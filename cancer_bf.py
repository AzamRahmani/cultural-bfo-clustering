import numpy as np

from bf_reusable import run_bf
from data_loader import load_standardized_dataset


X, y, X_scaled, _, _ = load_standardized_dataset("cancer")
result = run_bf(
    X_scaled,
    num_clusters=2,
    random_seed=42,
)

labels = result["labels"]
centers = result["centers"]
nonempty_cluster_count = len(np.unique(labels))
reported_numeric_values = np.array(
    [
        result["best_fitness"],
        result["silhouette"],
        result["empty_cluster_rejections"],
        nonempty_cluster_count,
    ],
    dtype=float,
)
all_reported_numeric_values_finite = np.all(
    np.isfinite(reported_numeric_values)
)

assert X.shape == (683, 9)
assert y.shape == (683,)
assert labels.shape == (683,)
assert centers.shape == (2, 9)
assert nonempty_cluster_count == 2
assert np.isfinite(result["best_fitness"])
assert np.isfinite(result["silhouette"])
assert all_reported_numeric_values_finite

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
print(f"best_fitness: {result['best_fitness']:.4f}")
print(f"silhouette: {result['silhouette']:.4f}")
print(
    "empty_cluster_rejections: "
    f"{result['empty_cluster_rejections']}"
)
print(f"nonempty cluster count: {nonempty_cluster_count}")
print(f"labels shape: {labels.shape}")
print(f"centers shape: {centers.shape}")
print(
    "all reported numeric values finite: "
    f"{all_reported_numeric_values_finite}"
)
