"""Analyze Project B results as fixed in PROJECT_B_ANALYSIS_PLAN.md."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_prototype_results import describe, holm_adjust, rank_biserial
from project_b_runs import RESULTS_DIR, SEEDS, _cell, metadata_path_for, read_results

CLUSTERING_REPRESENTATIONS = ("HAND", "RAW", "PCA16", "AE16")
CLUSTERING_ALGORITHMS = ("KMEANS", "BF", "CBF")
ACTIVITIES = (1, 2, 3, 4, 5, 6)
DETECTORS = ("AE", "PCA", "IF")
EDGE_MODELS = ("AE16_ENCODER", "PCA16")

CLUSTERING_METRICS = (
    "error_rate_percent", "ari", "nmi", "silhouette", "fitness",
    "clustering_runtime_seconds", "fitness_evaluation_count",
    "empty_cluster_rejections", "ae_final_training_loss", "ae_training_seconds",
)
ANOMALY_METRICS = ("roc_auc", "pr_auc", "detector_runtime_seconds")
BETTER = {
    "error_rate_percent": "lower", "ari": "higher", "nmi": "higher",
    "roc_auc": "higher", "pr_auc": "higher",
}

DESCRIPTIVE_COLUMNS = (
    "part", "context", "method", "metric", "successful_runs", "failed_runs",
    "mean", "std_population", "median", "minimum", "maximum", "q1", "q3",
    "iqr", "ci95_lower", "ci95_upper",
)
PAIRED_COLUMNS = (
    "family", "role", "metric", "context", "method_a", "method_b",
    "better_direction", "pairs", "excluded_seeds", "median_difference_a_minus_b",
    "a_lower_count", "a_higher_count", "tie_count", "wilcoxon_statistic",
    "p_value", "p_holm", "rank_biserial",
)


class ValidationError(ValueError):
    pass


# --- loading and validation ------------------------------------------------

def load_and_validate(results_dir, require_full=True):
    results_dir = Path(results_dir)
    data = {}
    expected = {
        "clustering": {(r, a, s) for r in CLUSTERING_REPRESENTATIONS
                       for a in CLUSTERING_ALGORITHMS for s in SEEDS},
        "anomaly": {(str(w), d, s) for w in ACTIVITIES for d in DETECTORS for s in SEEDS},
        "edge": set(EDGE_MODELS),
    }
    for part in ("clustering", "anomaly", "edge"):
        path = results_dir / f"{part}_results.csv"
        rows = read_results(path)  # verifies the fingerprint
        sidecar = json.loads(metadata_path_for(path).read_text(encoding="utf-8"))
        if require_full and not sidecar.get("full_protocol_design"):
            raise ValidationError(f"{path.name}: not a full protocol run")
        if sidecar["row_count"] != len(rows):
            raise ValidationError(f"{path.name}: row count differs from sidecar")
        if part == "clustering":
            keys = [(r["representation"], r["algorithm"], int(r["seed"])) for r in rows]
        elif part == "anomaly":
            keys = [(r["withheld_activity"], r["detector"], int(r["seed"])) for r in rows]
        else:
            keys = [r["model"] for r in rows]
        if len(set(keys)) != len(keys):
            raise ValidationError(f"{path.name}: duplicate records")
        if require_full and set(keys) != expected[part]:
            raise ValidationError(f"{path.name}: records differ from protocol Section 9")
        for row in rows:
            if row["run_status"] not in ("success", "failed"):
                raise ValidationError(f"{path.name}: unknown run status {row['run_status']!r}")
            if row["run_status"] == "failed" and not row["failure_description"]:
                raise ValidationError(f"{path.name}: failed run without description")
        data[part] = rows
    return data


# --- value selection -------------------------------------------------------

def seed_values(rows, metric, **conditions):
    selected = {}
    for row in rows:
        if all(row[key] == value for key, value in conditions.items()):
            if row["run_status"] == "success" and row[metric] != "":
                selected[int(row["seed"])] = float(row[metric])
    return selected


def failed_count(rows, **conditions):
    return sum(
        row["run_status"] == "failed"
        for row in rows
        if all(row[key] == value for key, value in conditions.items())
    )


def anomaly_all(rows, detector, metric):
    """Per-seed mean over the six activities; only seeds with all six."""
    per_activity = [
        seed_values(rows, metric, detector=detector, withheld_activity=str(a)) for a in ACTIVITIES
    ]
    seeds = set.intersection(*(set(values) for values in per_activity))
    return {seed: float(np.mean([values[seed] for values in per_activity])) for seed in sorted(seeds)}


# --- statistics -------------------------------------------------------------

def descriptive_row(part, context, method, metric, values, failures):
    summary = describe(list(values.values()))
    return {
        "part": part, "context": context, "method": method, "metric": metric,
        "successful_runs": len(values), "failed_runs": failures, **summary,
    }


def paired(values_a, values_b):
    seeds = sorted(set(values_a) & set(values_b))
    excluded = sorted((set(values_a) | set(values_b) | set(SEEDS)) - set(seeds))
    a = np.array([values_a[s] for s in seeds], dtype=float)
    b = np.array([values_b[s] for s in seeds], dtype=float)
    differences = a - b
    statistic = p_value = None
    if len(seeds) > 0 and not np.all(differences == 0):
        result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "pairs": len(seeds),
        "excluded_seeds": ";".join(str(s) for s in excluded),
        "median_difference_a_minus_b": float(np.median(differences)) if len(seeds) else None,
        "a_lower_count": int(np.sum(differences < 0)),
        "a_higher_count": int(np.sum(differences > 0)),
        "tie_count": int(np.sum(differences == 0)),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "rank_biserial": rank_biserial(differences) if len(seeds) else None,
    }


def comparison_plan(data):
    """(family, role, metric, context, method_a, values_a, method_b, values_b)."""
    clustering, anomaly = data["clustering"], data["anomaly"]

    def cv(representation, algorithm, metric):
        return seed_values(clustering, metric, representation=representation, algorithm=algorithm)

    plan = []
    for other in ("PCA16", "HAND", "RAW"):
        plan.append(("C", "confirmatory", "error_rate_percent", "KMEANS", "AE16",
                     cv("AE16", "KMEANS", "error_rate_percent"), other,
                     cv(other, "KMEANS", "error_rate_percent")))
    for other in ("PCA", "IF"):
        plan.append(("C", "confirmatory", "roc_auc", "ALL", "AE",
                     anomaly_all(anomaly, "AE", "roc_auc"), other, anomaly_all(anomaly, other, "roc_auc")))

    for algorithm in ("BF", "CBF"):
        for other in ("PCA16", "HAND", "RAW"):
            plan.append(("E1", "exploratory", "error_rate_percent", algorithm, "AE16",
                         cv("AE16", algorithm, "error_rate_percent"), other,
                         cv(other, algorithm, "error_rate_percent")))
    for metric in ("ari", "nmi"):
        for other in ("PCA16", "HAND", "RAW"):
            plan.append(("E2", "exploratory", metric, "KMEANS", "AE16",
                         cv("AE16", "KMEANS", metric), other, cv(other, "KMEANS", metric)))
    for activity in ACTIVITIES:
        for other in ("PCA", "IF"):
            plan.append(("E3", "exploratory", "roc_auc", str(activity), "AE",
                         seed_values(anomaly, "roc_auc", detector="AE", withheld_activity=str(activity)),
                         other,
                         seed_values(anomaly, "roc_auc", detector=other, withheld_activity=str(activity))))
    for other in ("PCA", "IF"):
        plan.append(("E3", "exploratory", "pr_auc", "ALL", "AE",
                     anomaly_all(anomaly, "AE", "pr_auc"), other, anomaly_all(anomaly, other, "pr_auc")))
    for representation in CLUSTERING_REPRESENTATIONS:
        for method_a, method_b in (("BF", "KMEANS"), ("CBF", "KMEANS"), ("CBF", "BF")):
            plan.append(("E4", "exploratory", "error_rate_percent", representation, method_a,
                         cv(representation, method_a, "error_rate_percent"), method_b,
                         cv(representation, method_b, "error_rate_percent")))
    return plan


def paired_rows(data):
    rows = []
    for family, role, metric, context, method_a, values_a, method_b, values_b in comparison_plan(data):
        rows.append({
            "family": family, "role": role, "metric": metric, "context": context,
            "method_a": method_a, "method_b": method_b, "better_direction": BETTER[metric],
            **paired(values_a, values_b), "p_holm": None,
        })
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        for row, adjusted in zip(members, holm_adjust([row["p_value"] for row in members])):
            row["p_holm"] = adjusted
    return rows


def clustering_descriptive(data):
    rows = []
    for representation in CLUSTERING_REPRESENTATIONS:
        for algorithm in CLUSTERING_ALGORITHMS:
            failures = failed_count(data["clustering"], representation=representation, algorithm=algorithm)
            for metric in CLUSTERING_METRICS:
                if metric.startswith("ae_") and representation != "AE16":
                    continue
                if metric in ("fitness_evaluation_count", "empty_cluster_rejections") and algorithm == "KMEANS":
                    continue
                values = seed_values(data["clustering"], metric, representation=representation, algorithm=algorithm)
                rows.append(descriptive_row("clustering", representation, algorithm, metric, values, failures))
    return rows


def anomaly_descriptive(data):
    rows = []
    for detector in DETECTORS:
        for activity in ACTIVITIES:
            failures = failed_count(data["anomaly"], detector=detector, withheld_activity=str(activity))
            for metric in ANOMALY_METRICS:
                values = seed_values(data["anomaly"], metric, detector=detector, withheld_activity=str(activity))
                rows.append(descriptive_row("anomaly", str(activity), detector, metric, values, failures))
        for metric in ANOMALY_METRICS:
            values = anomaly_all(data["anomaly"], detector, metric)
            rows.append(descriptive_row("anomaly", "ALL", detector, metric, values, len(SEEDS) - len(values)))
    return rows


# --- output -----------------------------------------------------------------

def write_rows(rows, columns, path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[column]) for column in columns])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "analysis")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="for tests only")
    arguments = parser.parse_args(argv)
    try:
        data = load_and_validate(arguments.results_dir, require_full=not arguments.allow_partial)
    except (ValidationError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    outputs = {
        "clustering_descriptive.csv": (clustering_descriptive(data), DESCRIPTIVE_COLUMNS),
        "anomaly_descriptive.csv": (anomaly_descriptive(data), DESCRIPTIVE_COLUMNS),
        "paired_tests.csv": (paired_rows(data), PAIRED_COLUMNS),
    }
    for name, (rows, columns) in outputs.items():
        try:
            write_rows(rows, columns, arguments.output_dir / name, arguments.overwrite)
        except FileExistsError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
    print(f"Wrote {', '.join(outputs)} to {arguments.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
