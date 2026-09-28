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
    "cancer": {"clusters": 2, "thesis_value": 2964.4},
}


def evaluate_pipeline(dataset_name, X, cluster_count, pipeline_name):
    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10,
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    inertia = float(kmeans.inertia_)
    squared_distance = within_cluster_sum_squared_distance(
        X, labels, centers
    )
    euclidean_distance = within_cluster_sum_euclidean_distance(
        X, labels, centers
    )
    silhouette = float(silhouette_score(X, labels))
    actual_clusters = int(np.unique(labels).size)

    assert np.isclose(inertia, squared_distance, rtol=1e-10, atol=1e-10), (
        f"{dataset_name} {pipeline_name}: sklearn inertia does not match "
        "the shared squared-distance metric."
    )
    assert np.all(
        np.isfinite(
            [inertia, squared_distance, euclidean_distance, silhouette]
        )
    ), f"{dataset_name} {pipeline_name}: a reported metric is not finite."
    assert actual_clusters == cluster_count, (
        f"{dataset_name} {pipeline_name}: expected {cluster_count} nonempty "
        f"clusters, got {actual_clusters}."
    )
    assert labels.shape == (X.shape[0],), (
        f"{dataset_name} {pipeline_name}: unexpected labels shape "
        f"{labels.shape}."
    )
    assert centers.shape == (cluster_count, X.shape[1]), (
        f"{dataset_name} {pipeline_name}: unexpected centers shape "
        f"{centers.shape}."
    )

    return {
        "inertia": inertia,
        "squared_distance": squared_distance,
        "euclidean_distance": euclidean_distance,
        "silhouette": silhouette,
        "actual_clusters": actual_clusters,
        "labels_shape": labels.shape,
        "centers_shape": centers.shape,
    }


for dataset_name, specification in DATASET_SPECS.items():
    X, _, X_scaled, _, _ = load_standardized_dataset(dataset_name)
    cluster_count = specification["clusters"]
    thesis_value = float(specification["thesis_value"])

    raw_results = evaluate_pipeline(
        dataset_name, X, cluster_count, "raw-data pipeline"
    )
    standardized_results = evaluate_pipeline(
        dataset_name, X_scaled, cluster_count, "standardized-data pipeline"
    )

    raw_difference = abs(raw_results["euclidean_distance"] - thesis_value)
    standardized_difference = abs(
        standardized_results["euclidean_distance"] - thesis_value
    )
    assert np.all(
        np.isfinite([thesis_value, raw_difference, standardized_difference])
    ), f"{dataset_name}: a thesis comparison value is not finite."

    if raw_difference < standardized_difference:
        closer_preprocessing = "raw data"
    elif standardized_difference < raw_difference:
        closer_preprocessing = "standardized data"
    else:
        closer_preprocessing = "equally close"

    print(f"Dataset: {dataset_name.capitalize()}")
    for pipeline_name, results in (
        ("Raw-data results", raw_results),
        ("Standardized-data results", standardized_results),
    ):
        print(f"{pipeline_name}:")
        print(f"  Requested clusters: {cluster_count}")
        print(f"  Actual nonempty clusters: {results['actual_clusters']}")
        print(f"  Labels shape: {results['labels_shape']}")
        print(f"  Centers shape: {results['centers_shape']}")
        print(f"  sklearn inertia: {results['inertia']:.12f}")
        print(
            "  Within-cluster sum squared distance: "
            f"{results['squared_distance']:.12f}"
        )
        print(
            "  Within-cluster sum Euclidean distance: "
            f"{results['euclidean_distance']:.12f}"
        )
        print(f"  Silhouette score: {results['silhouette']:.12f}")
    print(f"Thesis-reported K-Means distance value: {thesis_value}")
    print(f"Raw absolute difference: {raw_difference:.12f}")
    print(f"Standardized absolute difference: {standardized_difference:.12f}")
    print(f"Numerically closer current preprocessing: {closer_preprocessing}")
    print()

print("Warnings:")
print("- This comparison is diagnostic only.")
print("- A closer value does not verify thesis preprocessing.")
print(
    "- Differences may also result from dataset versions, initialization, "
    "candidate representation, objective definitions, or algorithm procedures."
)
print(
    "- Raw and standardized fitness values must not be interpreted as "
    "directly comparable measures of algorithm quality."
)
print("- Existing K-Means, BF, and CBF behavior was not changed.")
print("- These results do not establish thesis reproduction.")