import numpy as np

from clustering_evaluation import clustering_error_rate
from data_loader import load_standardized_dataset
from bf_reusable import run_bf
from cbf_reusable import run_cbf


DATASETS = (
    ("iris", 3, 6.67, 5.33),
    ("wine", 3, 12.36, 10.67),
    ("glass", 6, None, None),
    ("cancer", 2, 3.95, 3.37),
)


def report_result(
    dataset_name,
    method_name,
    result,
    X_scaled,
    y,
    requested_clusters,
    thesis_error_rate,
):
    labels = result["labels"]
    centers = result["centers"]
    sample_count = X_scaled.shape[0]
    feature_count = X_scaled.shape[1]

    actual_cluster_count = len(np.unique(labels))
    known_class_count = len(np.unique(y))
    evaluation = clustering_error_rate(y, labels)

    assert labels.shape == (sample_count,)
    assert centers.shape == (requested_clusters, feature_count)
    assert actual_cluster_count == requested_clusters
    assert requested_clusters == known_class_count
    assert np.isfinite(result["best_fitness"])
    assert np.isfinite(result["silhouette"])
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

    print(f"{method_name} | {dataset_name.capitalize()}")
    print(f"best_fitness: {result['best_fitness']:.12f}")
    print(f"silhouette: {result['silhouette']:.12f}")
    print(
        "empty_cluster_rejections: "
        f"{result['empty_cluster_rejections']}"
    )
    print(f"actual nonempty cluster count: {actual_cluster_count}")
    print(f"labels shape: {labels.shape}")
    print(f"centers shape: {centers.shape}")

    if method_name == "CBF":
        final_population_best = result[
            "final_population_best_fitness"
        ]
        centers_equal_belief = np.array_equal(
            centers,
            result["belief_best_bacterium"],
        )
        best_not_worse_than_population = (
            result["best_fitness"] <= final_population_best
        )
        assert np.isfinite(final_population_best)
        assert best_not_worse_than_population
        assert centers_equal_belief
        print(
            "final_population_best_fitness: "
            f"{final_population_best:.12f}"
        )
        print(
            "centers exactly equal belief_best_bacterium: "
            f"{centers_equal_belief}"
        )
        print(
            "best_fitness <= final_population_best_fitness: "
            f"{best_not_worse_than_population}"
        )

    print(
        "optimal cluster-to-class mapping: "
        f"{evaluation['mapping']}"
    )
    print("contingency matrix:")
    print(evaluation["contingency_matrix"])
    print(f"correct count: {evaluation['correct_count']}")
    print(f"incorrect count: {evaluation['incorrect_count']}")
    print(
        "accuracy percentage: "
        f"{evaluation['accuracy_percent']:.12f}"
    )
    print(
        "error-rate percentage: "
        f"{evaluation['error_rate_percent']:.12f}"
    )

    if thesis_error_rate is None:
        return

    difference = abs(
        evaluation["error_rate_percent"] - thesis_error_rate
    )
    reference_name = "BFOA" if method_name == "BF" else "proposed method"
    print(
        f"thesis {reference_name} error rate "
        f"(diagnostic reference): {thesis_error_rate:.2f}%"
    )
    print(
        "absolute percentage-point difference: "
        f"{difference:.12f}"
    )


def main():
    for (
        dataset_name,
        requested_clusters,
        thesis_bf_error_rate,
        thesis_cbf_error_rate,
    ) in DATASETS:
        _, y, X_scaled, _, _ = load_standardized_dataset(dataset_name)

        bf_result = run_bf(
            X_scaled,
            num_clusters=requested_clusters,
            random_seed=42,
        )
        cbf_result = run_cbf(
            X_scaled,
            num_clusters=requested_clusters,
            random_seed=42,
        )

        print(f"Dataset: {dataset_name.capitalize()}")
        report_result(
            dataset_name,
            "BF",
            bf_result,
            X_scaled,
            y,
            requested_clusters,
            thesis_bf_error_rate,
        )
        report_result(
            dataset_name,
            "CBF",
            cbf_result,
            X_scaled,
            y,
            requested_clusters,
            thesis_cbf_error_rate,
        )
        print()

        if dataset_name == "glass":
            print(
                "Thesis-reported BF/CBF error rates: "
                "not verified from current evidence"
            )
            print()

    print("Warnings:")
    print("- Known labels were used only after optimization.")
    print("- Labels were not used to select or alter BF or CBF solutions.")
    print("- Cluster identifiers do not automatically equal class identifiers.")
    print("- Optimal mapping is an external evaluation step.")
    print("- Seed-42 results are regression references, not universal targets.")
    print("- Thesis values are diagnostic references, not test targets.")
    print("- Numerical agreement does not prove methodological equivalence.")
    print(
        "- These results do not establish thesis reproduction "
        "or superiority."
    )


if __name__ == "__main__":
    main()