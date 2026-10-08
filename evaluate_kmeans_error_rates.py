import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from clustering_evaluation import clustering_error_rate
from data_loader import load_standardized_dataset


DATASETS = (
    ("iris", 3, 10.67),
    ("wine", 3, 16.85),
    ("glass", 6, None),
    ("cancer", 2, 5.86),
)


def evaluate_dataset(dataset_name, requested_clusters, thesis_error_rate):
    _, y, X_scaled, _, _ = load_standardized_dataset(dataset_name)

    kmeans = KMeans(
        n_clusters=requested_clusters,
        random_state=42,
        n_init=10,
    )
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    actual_cluster_count = len(np.unique(labels))
    known_class_count = len(np.unique(y))
    inertia = float(kmeans.inertia_)
    silhouette = float(silhouette_score(X_scaled, labels))

    evaluation = clustering_error_rate(y, labels)

    sample_count = X_scaled.shape[0]
    assert labels.shape == (sample_count,)
    assert requested_clusters == actual_cluster_count
    assert requested_clusters == known_class_count
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
    assert np.isfinite(inertia)
    assert np.isfinite(silhouette)
    assert 0.0 <= evaluation["error_rate_percent"] <= 100.0

    print(f"Dataset: {dataset_name.capitalize()}")
    print(f"X_scaled shape: {X_scaled.shape}")
    print(f"Requested cluster count: {requested_clusters}")
    print(f"Actual nonempty cluster count: {actual_cluster_count}")
    print(f"Known class count: {known_class_count}")
    print(f"Inertia: {inertia:.12f}")
    print(f"Silhouette: {silhouette:.12f}")
    print(f"Optimal cluster-to-class mapping: {evaluation['mapping']}")
    print("Contingency matrix:")
    print(evaluation["contingency_matrix"])
    print(f"Correct count: {evaluation['correct_count']}")
    print(f"Incorrect count: {evaluation['incorrect_count']}")
    print(f"Accuracy percentage: {evaluation['accuracy_percent']:.12f}")
    print(f"Error-rate percentage: {evaluation['error_rate_percent']:.12f}")

    if thesis_error_rate is None:
        print(
            "Thesis-reported K-Means error rate: "
            "not verified from current evidence"
        )
    else:
        difference = abs(
            evaluation["error_rate_percent"] - thesis_error_rate
        )
        print(
            "Thesis-reported K-Means error rate "
            f"(diagnostic reference): {thesis_error_rate:.2f}%"
        )
        print(
            "Absolute error-rate difference: "
            f"{difference:.12f} percentage points"
        )

    print()


def main():
    for dataset_name, requested_clusters, thesis_error_rate in DATASETS:
        evaluate_dataset(
            dataset_name,
            requested_clusters,
            thesis_error_rate,
        )

    print("Warnings:")
    print("- Known labels were used only after clustering.")
    print("- Cluster identifiers do not automatically equal class identifiers.")
    print("- Optimal label mapping is an external evaluation step.")
    print(
        "- Numerical agreement does not prove identical preprocessing "
        "or procedure."
    )
    print("- Thesis values are diagnostic references, not regression targets.")
    print("- These results do not establish thesis reproduction.")


if __name__ == "__main__":
    main()