"""Run the configurable repeated-seed BF/CBF prototype experiment."""

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DATASET_CLUSTER_COUNTS = {
    "iris": 3,
    "wine": 3,
    "glass": 6,
    "cancer": 2,
}
ALGORITHM_NAMES = ("BF", "CBF")
DEFAULT_SEEDS = tuple(range(30))

REQUIRED_COUNTER_FIELDS = (
    "accepted_movements",
    "reproduction_events",
    "elimination_events",
    "dispersed_bacteria_count",
    "fitness_evaluation_count",
    "empty_cluster_rejections",
)
RESULT_METRIC_FIELDS = (
    "best_fitness",
    "final_population_best_fitness",
    "silhouette",
    "error_rate_percent",
    "accuracy_percent",
    "correct_count",
    "incorrect_count",
    "nonempty_cluster_count",
    "empty_cluster_rejections",
    "runtime_seconds",
    "fitness_evaluation_count",
    "accepted_movements",
    "reproduction_events",
    "elimination_events",
    "dispersed_bacteria_count",
)


class MissingRequiredCounterError(RuntimeError):
    """A reusable algorithm result omitted a required counter."""


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Run selected configurations of the prototype experiment."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=tuple(DATASET_CLUSTER_COUNTS),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHM_NAMES,
        default=ALGORITHM_NAMES,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    arguments = parser.parse_args(argv)

    if not arguments.datasets:
        parser.error("at least one dataset must be selected")
    unknown_datasets = [
        dataset
        for dataset in arguments.datasets
        if dataset not in DATASET_CLUSTER_COUNTS
    ]
    if unknown_datasets:
        parser.error(f"unknown dataset(s): {unknown_datasets}")
    if len(set(arguments.datasets)) != len(arguments.datasets):
        parser.error("datasets must not contain duplicates")

    if not arguments.algorithms:
        parser.error("at least one algorithm must be selected")
    if len(set(arguments.algorithms)) != len(arguments.algorithms):
        parser.error("algorithms must not contain duplicates")

    if not arguments.seeds:
        parser.error("at least one seed must be selected")
    if any(seed < 0 for seed in arguments.seeds):
        parser.error("seeds must be nonnegative integers")
    if len(set(arguments.seeds)) != len(arguments.seeds):
        parser.error("seeds must not contain duplicates")

    if not arguments.dry_run:
        if arguments.output is None:
            parser.error("--output is required unless --dry-run is specified")
        for path in (arguments.output, metadata_path_for(arguments.output)):
            if path.exists() and not arguments.overwrite:
                parser.error(
                    f"output already exists; use --overwrite to replace it: "
                    f"{path}"
                )

    return arguments


def metadata_path_for(output_path):
    return output_path.with_name(output_path.stem + ".metadata.json")


def git_tree_is_clean():
    # Only tracked files count; untracked files cannot change committed code.
    repository_root = Path(__file__).resolve().parent
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not read the Git working-tree status.") from error
    return status.strip() == ""


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_run_metadata(path, run_metadata, overwrite):
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")


def planned_runs(arguments):
    return tuple(
        itertools.product(
            arguments.datasets,
            arguments.algorithms,
            arguments.seeds,
        )
    )


def print_plan(runs):
    for dataset, algorithm, seed in runs:
        print(f"dataset={dataset} algorithm={algorithm} seed={seed}")
    print(f"Total planned runs: {len(runs)}")


def get_reproducibility_metadata():
    repository_root = Path(__file__).resolve().parent
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not read the current Git commit hash.") from error
    if not git_commit:
        raise RuntimeError("Git returned an empty commit hash.")

    import matplotlib
    import numpy
    import scipy
    import sklearn

    return {
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "numpy_version": numpy.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
    }


def parameter_configuration(algorithm, dataset, seed):
    # Temporary prototype settings: explicitly record all reusable defaults.
    configuration = {
        "num_clusters": DATASET_CLUSTER_COUNTS[dataset],
        "random_seed": seed,
        "population_size": 20,
        "num_iterations": 50,
        "step_size": 0.1,
        "max_swim_steps": 3,
        "reproduction_interval": 10,
        "elimination_interval": 10,
        "elimination_probability": 0.10,
    }
    if algorithm == "CBF":
        configuration.update(
            {
                "cultural_influence": 0.05,
                "acceptance_ratio": 0.35,
            }
        )
    return configuration


def empty_result_row(dataset, algorithm, seed, configuration, metadata):
    row = {field: None for field in RESULT_METRIC_FIELDS}
    row.update(
        {
            "dataset": dataset,
            "algorithm": algorithm,
            "seed": seed,
            "run_status": "failed",
            "failure_description": "",
            "parameter_configuration": configuration,
            **metadata,
        }
    )
    return row


def describe_failure(error):
    message = " ".join(str(error).split())
    if not message:
        message = "No exception message was provided."
    description = f"{type(error).__name__}: {message}"
    if len(description) > 500:
        description = description[:497] + "..."
    return description


def execute_one_run(
    dataset,
    algorithm,
    seed,
    loaded_dataset,
    configuration,
    metadata,
    algorithm_runners,
    np,
    clustering_error_rate,
):
    X, y, X_scaled, _, _ = loaded_dataset
    requested_clusters = DATASET_CLUSTER_COUNTS[dataset]
    runner = algorithm_runners[algorithm]
    result = runner(
        X_scaled,
        num_clusters=requested_clusters,
        random_seed=seed,
        population_size=20,
        num_iterations=50,
        step_size=0.1,
        max_swim_steps=3,
        reproduction_interval=10,
        elimination_interval=10,
        elimination_probability=0.10,
        **(
            {
                "cultural_influence": 0.05,
                "acceptance_ratio": 0.35,
            }
            if algorithm == "CBF"
            else {}
        ),
    )

    missing_counters = [
        field for field in REQUIRED_COUNTER_FIELDS if field not in result
    ]
    if missing_counters:
        raise MissingRequiredCounterError(
            f"{algorithm} result for {dataset} seed {seed} is missing "
            f"required counter field(s): {missing_counters}"
        )
    unavailable_counters = [
        field for field in REQUIRED_COUNTER_FIELDS if result[field] is None
    ]
    if unavailable_counters:
        raise MissingRequiredCounterError(
            f"{algorithm} result for {dataset} seed {seed} has unavailable "
            f"required counter value(s): {unavailable_counters}"
        )

    labels = result["labels"]
    centers = result["centers"]
    if labels.shape != (len(X),):
        raise ValueError(
            f"Returned labels have shape {labels.shape}; expected {(len(X),)}."
        )
    expected_centers_shape = (requested_clusters, X_scaled.shape[1])
    if centers.shape != expected_centers_shape:
        raise ValueError(
            f"Returned centers have shape {centers.shape}; "
            f"expected {expected_centers_shape}."
        )
    nonempty_cluster_count = int(np.unique(labels).size)
    if nonempty_cluster_count != requested_clusters:
        raise ValueError(
            f"Run returned {nonempty_cluster_count} nonempty clusters; "
            f"expected {requested_clusters}."
        )

    if algorithm == "CBF":
        final_population_best_fitness = result[
            "final_population_best_fitness"
        ]
        if float(result["best_fitness"]) > float(
            final_population_best_fitness
        ):
            raise ValueError(
                "CBF best_fitness exceeds final_population_best_fitness."
            )
        if not np.array_equal(centers, result["belief_best_bacterium"]):
            raise ValueError(
                "CBF returned centers do not exactly match "
                "belief_best_bacterium."
            )
    else:
        final_population_best_fitness = None

    evaluation = clustering_error_rate(y, labels)
    row = empty_result_row(
        dataset,
        algorithm,
        seed,
        configuration,
        metadata,
    )
    row.update(
        {
            "best_fitness": result["best_fitness"],
            "final_population_best_fitness": final_population_best_fitness,
            "silhouette": result["silhouette"],
            "error_rate_percent": evaluation["error_rate_percent"],
            "accuracy_percent": evaluation["accuracy_percent"],
            "correct_count": evaluation["correct_count"],
            "incorrect_count": evaluation["incorrect_count"],
            "nonempty_cluster_count": nonempty_cluster_count,
            "empty_cluster_rejections": result["empty_cluster_rejections"],
            "runtime_seconds": result["runtime_seconds"],
            "fitness_evaluation_count": result["fitness_evaluation_count"],
            "accepted_movements": result["accepted_movements"],
            "reproduction_events": result["reproduction_events"],
            "elimination_events": result["elimination_events"],
            "dispersed_bacteria_count": result["dispersed_bacteria_count"],
            "run_status": "success",
            "failure_description": "",
        }
    )

    finite_fields = (
        "best_fitness",
        "silhouette",
        "error_rate_percent",
        "accuracy_percent",
        "runtime_seconds",
    )
    if algorithm == "CBF":
        finite_fields += ("final_population_best_fitness",)
    for field in finite_fields:
        if not math.isfinite(float(row[field])):
            raise ValueError(f"Successful run metric {field} is not finite.")
    return row


def execute_runs(runs, metadata):
    import numpy as np

    from bf_reusable import run_bf
    from cbf_reusable import run_cbf
    from clustering_evaluation import clustering_error_rate
    from data_loader import load_standardized_dataset

    algorithm_runners = {"BF": run_bf, "CBF": run_cbf}
    loaded_datasets = {}
    dataset_load_errors = {}
    rows = []

    for dataset, algorithm, seed in runs:
        configuration = parameter_configuration(algorithm, dataset, seed)
        if dataset not in loaded_datasets and dataset not in dataset_load_errors:
            try:
                loaded_datasets[dataset] = load_standardized_dataset(dataset)
            except Exception as error:
                dataset_load_errors[dataset] = error

        if dataset in dataset_load_errors:
            error = dataset_load_errors[dataset]
            row = empty_result_row(
                dataset,
                algorithm,
                seed,
                configuration,
                metadata,
            )
            row["failure_description"] = describe_failure(error)
        else:
            try:
                row = execute_one_run(
                    dataset,
                    algorithm,
                    seed,
                    loaded_datasets[dataset],
                    configuration,
                    metadata,
                    algorithm_runners,
                    np,
                    clustering_error_rate,
                )
            except MissingRequiredCounterError:
                raise
            except Exception as error:
                row = empty_result_row(
                    dataset,
                    algorithm,
                    seed,
                    configuration,
                    metadata,
                )
                row["failure_description"] = describe_failure(error)

        rows.append(row)
        if row["run_status"] == "success":
            suffix = (
                " (CBF archived-best consistency verified)"
                if algorithm == "CBF"
                else ""
            )
            print(
                f"SUCCESS dataset={dataset} algorithm={algorithm} "
                f"seed={seed}{suffix}"
            )
        else:
            print(
                f"FAILED dataset={dataset} algorithm={algorithm} seed={seed}: "
                f"{row['failure_description']}",
                file=sys.stderr,
            )
    return rows


def main(argv=None):
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = parse_arguments(command_arguments)
    runs = planned_runs(arguments)
    if arguments.dry_run:
        print_plan(runs)
        return 0

    try:
        tree_clean = git_tree_is_clean()
        if not tree_clean and not arguments.allow_dirty_tree:
            print(
                "ERROR: Tracked files have uncommitted changes. Commit them "
                "first, or use --allow-dirty-tree for a non-final run.",
                file=sys.stderr,
            )
            return 1

        from experiment_results import write_experiment_results

        started_at_utc = utc_timestamp()
        metadata = get_reproducibility_metadata()
        rows = execute_runs(runs, metadata)
        completed_at_utc = utc_timestamp()
    except MissingRequiredCounterError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"ERROR: Could not prepare experiment execution: "
            f"{describe_failure(error)}",
            file=sys.stderr,
        )
        return 1

    try:
        write_experiment_results(
            rows,
            arguments.output,
            overwrite=arguments.overwrite,
        )
    except Exception as error:
        print(
            f"ERROR: Could not write experiment results: "
            f"{describe_failure(error)}",
            file=sys.stderr,
        )
        return 1

    failed_runs = sum(row["run_status"] == "failed" for row in rows)
    metadata_path = metadata_path_for(arguments.output)
    run_metadata = {
        **metadata,
        "git_tree_clean": tree_clean,
        "allow_dirty_tree": arguments.allow_dirty_tree,
        "operating_system": platform.platform(),
        "command": [sys.executable, Path(__file__).name, *command_arguments],
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "results_file": arguments.output.name,
        "row_count": len(rows),
        "failed_row_count": failed_runs,
        "results_sha256": hashlib.sha256(
            arguments.output.read_bytes()
        ).hexdigest(),
    }
    try:
        write_run_metadata(metadata_path, run_metadata, arguments.overwrite)
    except Exception as error:
        print(
            f"ERROR: Could not write run metadata: "
            f"{describe_failure(error)}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Wrote {len(rows)} experiment result rows to {arguments.output} "
        f"({failed_runs} failed) and run metadata to {metadata_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
