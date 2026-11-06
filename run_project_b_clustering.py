"""Project B clustering experiment (PROJECT_B_PROTOCOL.md Sections 6, 9-10).

For every seed: train the autoencoder on the training windows, represent the
2,947 test windows as HAND, RAW, PCA-16, and AE-16, and cluster each with
K-Means, BF, and CBF (k = 6). Labels are used only for evaluation afterwards.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from bf_reusable import run_bf
from cbf_reusable import run_cbf
from clustering_evaluation import clustering_error_rate
from clustering_metrics import within_cluster_sum_squared_distance
from har_autoencoder import TrainingConfig, encode, train_autoencoder
from har_preprocessing import prepare_standardized
from project_b_runs import (
    NUM_CLUSTERS,
    RESULTS_DIR,
    SEEDS,
    check_run_preconditions,
    describe_failure,
    environment_metadata,
    run_command,
    utc_timestamp,
    write_results,
)

REPRESENTATIONS = ("HAND", "RAW", "PCA16", "AE16")
ALGORITHMS = ("KMEANS", "BF", "CBF")
LATENT_DIM = 16
DEFAULT_OUTPUT = RESULTS_DIR / "clustering_results.csv"

# Project A reusable defaults (EXPERIMENT_PROTOCOL.md Section 6), unchanged.
BF_PARAMETERS = {
    "population_size": 20,
    "num_iterations": 50,
    "step_size": 0.1,
    "max_swim_steps": 3,
    "reproduction_interval": 10,
    "elimination_interval": 10,
    "elimination_probability": 0.10,
}
CBF_PARAMETERS = {**BF_PARAMETERS, "cultural_influence": 0.05, "acceptance_ratio": 0.35}
KMEANS_PARAMETERS = {"n_init": 10}

COLUMNS = (
    "representation",
    "representation_dim",
    "algorithm",
    "seed",
    "run_status",
    "failure_description",
    "error_rate_percent",
    "accuracy_percent",
    "correct_count",
    "incorrect_count",
    "ari",
    "nmi",
    "silhouette",
    "fitness",
    "nonempty_cluster_count",
    "clustering_runtime_seconds",
    "fitness_evaluation_count",
    "empty_cluster_rejections",
    "ae_final_training_loss",
    "ae_training_seconds",
    "parameter_configuration",
)


def algorithm_configuration(algorithm, seed):
    if algorithm == "KMEANS":
        return {"n_clusters": NUM_CLUSTERS, "random_state": seed, **KMEANS_PARAMETERS}
    parameters = BF_PARAMETERS if algorithm == "BF" else CBF_PARAMETERS
    return {"num_clusters": NUM_CLUSTERS, "random_seed": seed, **parameters}


def empty_row(representation, dimension, algorithm, seed, configuration):
    row = {column: None for column in COLUMNS}
    row.update(
        representation=representation,
        representation_dim=dimension,
        algorithm=algorithm,
        seed=seed,
        run_status="failed",
        failure_description="",
        parameter_configuration=configuration,
    )
    return row


def cluster(algorithm, X, seed):
    """Return (labels, centers, extras) for one clustering run."""
    configuration = algorithm_configuration(algorithm, seed)
    start = time.perf_counter()
    if algorithm == "KMEANS":
        model = KMeans(**configuration).fit(X)
        runtime = time.perf_counter() - start
        return model.labels_.astype(np.int64), model.cluster_centers_, {
            "clustering_runtime_seconds": runtime,
        }

    runner = run_bf if algorithm == "BF" else run_cbf
    result = runner(X, **configuration)
    labels = np.asarray(result["labels"], dtype=np.int64)
    centers = np.asarray(result["centers"])
    if algorithm == "CBF":
        # Archived-best consistency, as checked in Project A.
        if float(result["best_fitness"]) > float(result["final_population_best_fitness"]):
            raise ValueError("CBF best_fitness exceeds final_population_best_fitness.")
        if not np.array_equal(centers, result["belief_best_bacterium"]):
            raise ValueError("CBF centers do not match belief_best_bacterium.")
    return labels, centers, {
        "clustering_runtime_seconds": float(result["runtime_seconds"]),
        "fitness_evaluation_count": int(result["fitness_evaluation_count"]),
        "empty_cluster_rejections": int(result["empty_cluster_rejections"]),
    }


def evaluate_run(representation, algorithm, seed, X, y, ae_history):
    configuration = algorithm_configuration(algorithm, seed)
    row = empty_row(representation, X.shape[1], algorithm, seed, configuration)
    if ae_history is not None:
        row["ae_final_training_loss"] = ae_history["final_training_loss"]
        row["ae_training_seconds"] = ae_history["training_seconds"]

    labels, centers, extras = cluster(algorithm, X, seed)
    if labels.shape != (len(X),) or centers.shape != (NUM_CLUSTERS, X.shape[1]):
        raise ValueError(f"unexpected output shapes {labels.shape}, {centers.shape}")
    nonempty = int(np.unique(labels).size)
    if nonempty != NUM_CLUSTERS:
        raise ValueError(f"{nonempty} nonempty clusters; expected {NUM_CLUSTERS}")

    evaluation = clustering_error_rate(y, labels)
    row.update(extras)
    row.update(
        error_rate_percent=float(evaluation["error_rate_percent"]),
        accuracy_percent=float(evaluation["accuracy_percent"]),
        correct_count=int(evaluation["correct_count"]),
        incorrect_count=int(evaluation["incorrect_count"]),
        ari=float(adjusted_rand_score(y, labels)),
        nmi=float(normalized_mutual_info_score(y, labels)),
        silhouette=float(silhouette_score(X, labels)),
        fitness=within_cluster_sum_squared_distance(X, labels, centers),
        nonempty_cluster_count=nonempty,
        run_status="success",
    )
    for column in ("error_rate_percent", "ari", "nmi", "silhouette", "fitness"):
        if not np.isfinite(row[column]):
            raise ValueError(f"{column} is not finite")
    return row


def static_representations(prepared):
    """Seed-independent test representations, all fitted on training windows."""
    pca = PCA(n_components=LATENT_DIM, svd_solver="full").fit(prepared["fit"]["raw_flat"])
    return {
        "HAND": prepared["test"]["hand"],
        "RAW": prepared["test"]["raw_flat"],
        "PCA16": pca.transform(prepared["test"]["raw_flat"]),
    }


def run_experiment(splits, seeds, representations, algorithms, ae_config, log=print):
    prepared = prepare_standardized(splits)
    y_test = prepared["test"]["y"]
    static = static_representations(prepared)
    rows = []

    for seed in seeds:
        test_representations = {name: static[name] for name in static if name in representations}
        ae_history, ae_error = None, None
        if "AE16" in representations:
            try:
                model, ae_history = train_autoencoder(prepared["fit"]["raw"], seed, ae_config)
                test_representations["AE16"] = encode(model, prepared["test"]["raw"])
            except Exception as error:
                ae_error = error
                log(f"FAILED autoencoder seed={seed}: {describe_failure(error)}")

        for representation in representations:
            for algorithm in algorithms:
                configuration = algorithm_configuration(algorithm, seed)
                history = ae_history if representation == "AE16" else None
                if representation == "AE16" and ae_error is not None:
                    row = empty_row(representation, LATENT_DIM, algorithm, seed, configuration)
                    row["failure_description"] = "autoencoder training failed: " + describe_failure(ae_error)
                else:
                    X = test_representations[representation]
                    try:
                        row = evaluate_run(representation, algorithm, seed, X, y_test, history)
                    except Exception as error:
                        row = empty_row(representation, X.shape[1], algorithm, seed, configuration)
                        if history is not None:
                            row["ae_final_training_loss"] = history["final_training_loss"]
                            row["ae_training_seconds"] = history["training_seconds"]
                        row["failure_description"] = describe_failure(error)
                rows.append(row)
                status = "SUCCESS" if row["run_status"] == "success" else "FAILED"
                detail = "" if status == "SUCCESS" else f": {row['failure_description']}"
                log(f"{status} seed={seed} representation={representation} algorithm={algorithm}{detail}")
    return rows


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=REPRESENTATIONS)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument("--ae-epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    arguments = parser.parse_args(argv)
    full_design = (
        tuple(arguments.seeds) == SEEDS
        and tuple(arguments.representations) == REPRESENTATIONS
        and tuple(arguments.algorithms) == ALGORITHMS
        and arguments.ae_epochs == TrainingConfig().epochs
    )
    if not full_design and arguments.output.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error("a partial or modified design must not write to the protocol output path")
    for name, values in (("seeds", arguments.seeds), ("representations", arguments.representations),
                         ("algorithms", arguments.algorithms)):
        if len(set(values)) != len(values):
            parser.error(f"{name} must not contain duplicates")
    arguments.full_design = full_design
    return arguments


def main(argv=None, splits=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    arguments = parse_arguments(argv)
    tree_clean, problems = check_run_preconditions(arguments.allow_dirty_tree)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    if splits is None:
        from har_data import load_har

        splits = load_har()

    ae_config = TrainingConfig(epochs=arguments.ae_epochs)
    metadata = environment_metadata()
    started = utc_timestamp()
    rows = run_experiment(
        splits, arguments.seeds, arguments.representations, arguments.algorithms,
        ae_config, log=lambda message: print(message, flush=True),
    )
    metadata.update(
        experiment="clustering",
        full_protocol_design=arguments.full_design,
        git_tree_clean=tree_clean,
        allow_dirty_tree=arguments.allow_dirty_tree,
        command=run_command(Path(__file__).name, argv),
        started_at_utc=started,
        completed_at_utc=utc_timestamp(),
        autoencoder_config=ae_config.as_dict(),
        seeds=list(arguments.seeds),
        representations=list(arguments.representations),
        algorithms=list(arguments.algorithms),
    )
    summary = write_results(rows, COLUMNS, arguments.output, metadata, arguments.overwrite)
    print(f"Wrote {summary['row_count']} rows ({summary['failed_row_count']} failed) to {arguments.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
