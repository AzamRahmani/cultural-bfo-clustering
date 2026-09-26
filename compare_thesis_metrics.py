import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from clustering_metrics import (
    within_cluster_sum_euclidean_distance,
    within_cluster_sum_squared_distance,
)
from data_loader import load_standardized_dataset


DATASET_SPECS = {
    "iris": {"clusters": 3, "thesis_value": 97.33},
    "wine": {"clusters": 3, "thesis_value": 16555.7},
    "glass": {"clusters": 6, "thesis_value": None},
    "cancer": {"clusters": 2, "thesis_value": 2964.4},
}


for dataset_name, specification in DATASET_SPECS.items():
    X, _, X_scaled, _, _ = load_standardized_dataset(dataset_name)
    requested_clusters = specification["clusters"]

    kmeans = KMeans(
        n_clusters=requested_clusters,
        random_state=42,
        n_init=10,
    )
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_

    inertia = float(kmeans.inertia_)
    squared_distance = within_cluster_sum_squared_distance(
        X_scaled, labels, centers
    )
    euclidean_distance = within_cluster_sum_euclidean_distance(
        X_scaled, labels, centers
    )
    silhouette = float(silhouette_score(X_scaled, labels))
    actual_clusters = int(np.unique(labels).size)

    assert np.isclose(inertia, squared_distance, rtol=1e-10, atol=1e-10), (
        f"{dataset_name}: sklearn inertia does not match the squared-distance metric."
    )
    assert np.all(
        np.isfinite(
            [inertia, squared_distance, euclidean_distance, silhouette]
        )
    ), f"{dataset_name}: a reported metric is not finite."
    assert actual_clusters == requested_clusters, (
        f"{dataset_name}: expected {requested_clusters} nonempty clusters, "
        f"got {actual_clusters}."
    )
    assert labels.shape == (X_scaled.shape[0],), (
        f"{dataset_name}: unexpected labels shape {labels.shape}."
    )
    assert centers.shape == (requested_clusters, X_scaled.shape[1]), (
        f"{dataset_name}: unexpected centers shape {centers.shape}."
    )

    print(f"Dataset: {dataset_name.capitalize()}")
    print(f"Standardized X shape: {X_scaled.shape}")
    print(f"Requested clusters: {requested_clusters}")
    print(f"Actual nonempty clusters: {actual_clusters}")
    print(f"Labels shape: {labels.shape}")
    print(f"Centers shape: {centers.shape}")
    print(f"sklearn inertia: {inertia:.12f}")
    print(f"Shared squared-distance result: {squared_distance:.12f}")
    print(f"Ordinary Euclidean-distance result: {euclidean_distance:.12f}")
    print(f"Silhouette: {silhouette:.12f}")

    thesis_value = specification["thesis_value"]
    if thesis_value is None:
        print(
            "Thesis-reported K-Means value: "
            "not verified from current evidence"
        )
    else:
        difference = euclidean_distance - thesis_value
        assert np.isfinite(difference), (
            f"{dataset_name}: the thesis-value difference is not finite."
        )
        print(f"Thesis-reported K-Means value: {thesis_value}")
        print(
            "Difference (ordinary minus thesis; diagnostic only): "
            f"{difference:.12f}"
        )
    print()

print("Warnings:")
print("- The thesis-compatible metric is used only for diagnostic reporting.")
print("- Existing K-Means, BF, and CBF optimization behavior was not changed.")
print(
    "- Agreement or disagreement with thesis values may also depend on "
    "preprocessing, initialization, parameters, dataset versions, and procedure."
)
print("- These results do not establish thesis reproduction.")
