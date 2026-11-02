"""Analyze the experiment v1 results exactly as predeclared in ANALYSIS_PLAN.md."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from experiment_results import read_experiment_results
from run_prototype_experiments import (
    ALGORITHM_NAMES,
    DATASET_CLUSTER_COUNTS,
    DEFAULT_SEEDS,
    metadata_path_for,
)

PRIMARY_METRIC = "best_fitness"
METRIC_DIRECTIONS = {
    "best_fitness": "lower",
    "silhouette": "higher",
    "error_rate_percent": "lower",
    "runtime_seconds": "lower",
    "fitness_evaluation_count": "lower",
    "empty_cluster_rejections": "lower",
}
DESCRIPTIVE_COLUMNS = (
    "dataset",
    "algorithm",
    "metric",
    "successful_runs",
    "failed_runs",
    "mean",
    "std_population",
    "median",
    "minimum",
    "maximum",
    "q1",
    "q3",
    "iqr",
    "ci95_lower",
    "ci95_upper",
)
PAIRED_COLUMNS = (
    "dataset",
    "metric",
    "role",
    "better_direction",
    "pairs",
    "excluded_seeds",
    "median_difference_cbf_minus_bf",
    "cbf_lower_count",
    "cbf_higher_count",
    "tie_count",
    "wilcoxon_statistic",
    "p_value",
    "p_value_holm",
    "rank_biserial",
)


def validate_result_set(rows, results_path):
    """Refuse analysis unless the file and record set match Protocol Section 15."""
    metadata_path = metadata_path_for(results_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(results_path.read_bytes()).hexdigest()
    if fingerprint != metadata["results_sha256"]:
        raise ValueError("Results file does not match its metadata fingerprint.")
    if len(rows) != metadata["row_count"]:
        raise ValueError("Row count does not match the run metadata.")

    expected_keys = {
        (dataset, algorithm, seed)
        for dataset in DATASET_CLUSTER_COUNTS
        for algorithm in ALGORITHM_NAMES
        for seed in DEFAULT_SEEDS
    }
    keys = [(row["dataset"], row["algorithm"], int(row["seed"])) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Results do not contain every predefined run exactly once.")


def describe(values):
    values = np.asarray(values, dtype=float)
    count = values.size
    if count == 0:
        return {column: None for column in DESCRIPTIVE_COLUMNS[5:]}

    mean = float(np.mean(values))
    q1, q3 = np.percentile(values, [25, 75], method="linear")
    if count < 2:
        ci_lower = ci_upper = None
    else:
        sample_std = float(np.std(values, ddof=1))
        half_width = (
            float(stats.t.ppf(0.975, count - 1)) * sample_std / math.sqrt(count)
        )
        ci_lower, ci_upper = mean - half_width, mean + half_width
    return {
        "mean": mean,
        "std_population": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
    }


def rank_biserial(differences):
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    rank_plus = float(ranks[nonzero > 0].sum())
    rank_minus = float(ranks[nonzero < 0].sum())
    return (rank_plus - rank_minus) / (rank_plus + rank_minus)


def paired_comparison(bf_values, cbf_values):
    bf_values = np.asarray(bf_values, dtype=float)
    cbf_values = np.asarray(cbf_values, dtype=float)
    differences = cbf_values - bf_values
    if np.all(differences == 0):
        statistic = p_value = None
    else:
        result = stats.wilcoxon(
            cbf_values,
            bf_values,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    return {
        "pairs": int(differences.size),
        "median_difference_cbf_minus_bf": float(np.median(differences)),
        "cbf_lower_count": int(np.sum(differences < 0)),
        "cbf_higher_count": int(np.sum(differences > 0)),
        "tie_count": int(np.sum(differences == 0)),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "rank_biserial": rank_biserial(differences),
    }


def holm_adjust(p_values):
    """Holm-Bonferroni adjustment; None marks a comparison without a test."""
    tested = sorted(
        (p_value, index)
        for index, p_value in enumerate(p_values)
        if p_value is not None
    )
    adjusted = [None] * len(p_values)
    running_maximum = 0.0
    for rank, (p_value, index) in enumerate(tested):
        running_maximum = max(
            running_maximum,
            min(1.0, (len(tested) - rank) * p_value),
        )
        adjusted[index] = running_maximum
    return adjusted


def successful_values(rows, dataset, algorithm, metric):
    return {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["dataset"] == dataset
        and row["algorithm"] == algorithm
        and row["run_status"] == "success"
    }


def descriptive_rows(rows):
    summary_rows = []
    for dataset in DATASET_CLUSTER_COUNTS:
        for algorithm in ALGORITHM_NAMES:
            failed_runs = sum(
                row["dataset"] == dataset
                and row["algorithm"] == algorithm
                and row["run_status"] == "failed"
                for row in rows
            )
            for metric in METRIC_DIRECTIONS:
                values = successful_values(rows, dataset, algorithm, metric)
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "metric": metric,
                        "successful_runs": len(values),
                        "failed_runs": failed_runs,
                        **describe(list(values.values())),
                    }
                )
    return summary_rows


def paired_rows(rows):
    comparison_rows = []
    for metric, direction in METRIC_DIRECTIONS.items():
        metric_rows = []
        for dataset in DATASET_CLUSTER_COUNTS:
            bf = successful_values(rows, dataset, "BF", metric)
            cbf = successful_values(rows, dataset, "CBF", metric)
            paired_seeds = sorted(set(bf) & set(cbf))
            excluded_seeds = sorted(set(DEFAULT_SEEDS) - set(paired_seeds))
            metric_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "role": (
                        "primary" if metric == PRIMARY_METRIC else "secondary"
                    ),
                    "better_direction": direction,
                    "excluded_seeds": ";".join(map(str, excluded_seeds)),
                    **paired_comparison(
                        [bf[seed] for seed in paired_seeds],
                        [cbf[seed] for seed in paired_seeds],
                    ),
                }
            )
        adjusted = holm_adjust([row["p_value"] for row in metric_rows])
        for row, p_value_holm in zip(metric_rows, adjusted):
            row["p_value_holm"] = p_value_holm
        comparison_rows.extend(metric_rows)
    return comparison_rows


def write_rows(rows, columns, path, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                ["" if row[column] is None else row[column] for column in columns]
            )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze the experiment v1 results as predeclared."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results") / "prototype_30_seed_results.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def output_paths(output_dir):
    return (
        output_dir / "prototype_30_seed_descriptive.csv",
        output_dir / "prototype_30_seed_paired_tests.csv",
    )


def main(argv=None):
    arguments = parse_arguments(argv)
    descriptive_path, paired_path = output_paths(arguments.output_dir)
    existing = [
        path for path in (descriptive_path, paired_path) if path.exists()
    ]
    if existing and not arguments.overwrite:
        print(
            f"ERROR: output already exists; use --overwrite: {existing}",
            file=sys.stderr,
        )
        return 1

    try:
        rows = read_experiment_results(arguments.results)
        validate_result_set(rows, arguments.results)
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: Results failed validation: {error}", file=sys.stderr)
        return 1

    write_rows(
        descriptive_rows(rows),
        DESCRIPTIVE_COLUMNS,
        descriptive_path,
        arguments.overwrite,
    )
    write_rows(paired_rows(rows), PAIRED_COLUMNS, paired_path, arguments.overwrite)
    print(f"Wrote {descriptive_path} and {paired_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
