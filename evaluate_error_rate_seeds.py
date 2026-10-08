import numpy as np

from clustering_evaluation import clustering_error_rate
from data_loader import load_standardized_dataset
from bf_reusable import run_bf
from cbf_reusable import run_cbf


DATASETS = (
    ("iris", 3),
    ("wine", 3),
    ("glass", 6),
    ("cancer", 2),
)
SEEDS = (0, 1, 2, 3, 4)
ALGORITHMS = ("BF", "CBF")


def record_run(
    dataset_name,
    algorithm_name,
    seed,
    result,
    X_scaled,
    y,
    requested_clusters,
):
    labels = result["labels"]
    centers = result["centers"]
    sample_count, feature_count = X_scaled.shape

    actual_cluster_count = len(np.unique(labels))
    known_class_count = len(np.unique(y))
    evaluation = clustering_error_rate(y, labels)

    assert labels.shape == (sample_count,)
    assert centers.shape == (requested_clusters, feature_count)
    assert actual_cluster_count == requested_clusters
    assert requested_clusters == known_class_count
    assert np.isfinite(result["best_fitness"])
    assert np.isfinite(result["silhouette"])
    assert np.isfinite(evaluation["accuracy_percent"])
    assert np.isfinite(evaluation["error_rate_percent"])
    assert (
        evaluation["correct_count"]
        + evaluation["incorrect_count"]
        == sample_count
    )
    assert np.isclose(
        evaluation["accuracy_percent"]
        + evaluation["error_rate_percent"],
        100.0,
    )
    assert 0.0 <= evaluation["error_rate_percent"] <= 100.0

    record = {
        "seed": seed,
        "best_fitness": float(result["best_fitness"]),
        "silhouette": float(result["silhouette"]),
        "error_rate_percent": float(
            evaluation["error_rate_percent"]
        ),
        "accuracy_percent": float(
            evaluation["accuracy_percent"]
        ),
        "correct_count": evaluation["correct_count"],
        "incorrect_count": evaluation["incorrect_count"],
        "empty_cluster_rejections": result[
            "empty_cluster_rejections"
        ],
        "actual_cluster_count": actual_cluster_count,
        "mapping": evaluation["mapping"],
    }

    if algorithm_name == "CBF":
        final_population_best = float(
            result["final_population_best_fitness"]
        )
        centers_match_belief = np.array_equal(
            centers,
            result["belief_best_bacterium"],
        )
        assert np.isfinite(final_population_best)
        assert result["best_fitness"] <= final_population_best
        assert centers_match_belief
        record["final_population_best_fitness"] = (
            final_population_best
        )
        record["centers_match_belief_best_bacterium"] = (
            centers_match_belief
        )

    return record


def print_seed_result(dataset_name, algorithm_name, record):
    print(
        f"Seed {record['seed']}: "
        f"best_fitness={record['best_fitness']:.12f}, "
        f"silhouette={record['silhouette']:.12f}, "
        f"error_rate_percent={record['error_rate_percent']:.12f}, "
        f"accuracy_percent={record['accuracy_percent']:.12f}, "
        f"correct_count={record['correct_count']}, "
        f"incorrect_count={record['incorrect_count']}, "
        "empty_cluster_rejections="
        f"{record['empty_cluster_rejections']}, "
        f"actual_nonempty_clusters={record['actual_cluster_count']}, "
        f"mapping={record['mapping']}"
    )
    if algorithm_name == "CBF":
        print(
            "  final_population_best_fitness="
            f"{record['final_population_best_fitness']:.12f}, "
            "centers_match_belief_best_bacterium="
            f"{record['centers_match_belief_best_bacterium']}"
        )


def print_summary(dataset_name, algorithm_name, records):
    error_rates = np.array(
        [record["error_rate_percent"] for record in records],
        dtype=float,
    )
    accuracies = np.array(
        [record["accuracy_percent"] for record in records],
        dtype=float,
    )
    fitness_values = np.array(
        [record["best_fitness"] for record in records],
        dtype=float,
    )
    silhouettes = np.array(
        [record["silhouette"] for record in records],
        dtype=float,
    )

    print(f"{algorithm_name} aggregate statistics")
    print(f"Mean error rate: {np.mean(error_rates):.12f}%")
    print(
        "Population error-rate standard deviation (ddof=0): "
        f"{np.std(error_rates, ddof=0):.12f} percentage points"
    )
    print(f"Median error rate: {np.median(error_rates):.12f}%")
    print(f"Minimum error rate: {np.min(error_rates):.12f}%")
    print(f"Maximum error rate: {np.max(error_rates):.12f}%")
    print(f"Mean accuracy: {np.mean(accuracies):.12f}%")
    print(f"Mean fitness: {np.mean(fitness_values):.12f}")
    print(f"Mean silhouette: {np.mean(silhouettes):.12f}")
    print(
        "Nonempty cluster counts by seed: "
        f"{[record['actual_cluster_count'] for record in records]}"
    )
    print(
        "Empty-cluster rejection counts by seed: "
        f"{[record['empty_cluster_rejections'] for record in records]}"
    )


def main():
    for dataset_name, requested_clusters in DATASETS:
        _, y, X_scaled, _, _ = load_standardized_dataset(dataset_name)
        results = {algorithm: [] for algorithm in ALGORITHMS}

        print(f"Dataset: {dataset_name.capitalize()}")
        print(f"X_scaled shape: {X_scaled.shape}")
        print(f"Requested cluster count: {requested_clusters}")

        for seed in SEEDS:
            bf_result = run_bf(
                X_scaled,
                num_clusters=requested_clusters,
                random_seed=seed,
            )
            cbf_result = run_cbf(
                X_scaled,
                num_clusters=requested_clusters,
                random_seed=seed,
            )

            for algorithm_name, result in (
                ("BF", bf_result),
                ("CBF", cbf_result),
            ):
                record = record_run(
                    dataset_name,
                    algorithm_name,
                    seed,
                    result,
                    X_scaled,
                    y,
                    requested_clusters,
                )
                results[algorithm_name].append(record)
                print_seed_result(
                    dataset_name,
                    algorithm_name,
                    record,
                )

        for algorithm_name in ALGORITHMS:
            print_summary(
                dataset_name,
                algorithm_name,
                results[algorithm_name],
            )
        print()

    print("Warnings:")
    print("- Known labels were used only after optimization.")
    print("- Labels were not used to select or modify solutions.")
    print("- Five seeds are an exploratory prototype evaluation.")
    print("- Five seeds do not establish statistical significance.")
    print("- Error rate depends on external cluster-to-class assignment.")
    print("- Results do not prove BF or CBF superiority.")
    print("- Results do not establish thesis reproduction.")


if __name__ == "__main__":
    main()