import numpy as np

from cbf_reusable import run_cbf
from data_loader import load_standardized_dataset


X, _, X_scaled, _, _ = load_standardized_dataset("glass")
result = run_cbf(
    X_scaled,
    num_clusters=6,
    random_seed=42,
)

labels = result["labels"]
nonempty_cluster_count = len(np.unique(labels))
assert result["best_fitness"] <= result[
    "final_population_best_fitness"
]
assert len(np.unique(labels)) == 6

print(f"X.shape: {X.shape}")
print(f"best_fitness: {result['best_fitness']:.4f}")
print(
    "final_population_best_fitness: "
    f"{result['final_population_best_fitness']:.4f}"
)
print(f"silhouette: {result['silhouette']:.4f}")
print(
    "empty_cluster_rejections: "
    f"{result['empty_cluster_rejections']}"
)
print(f"nonempty cluster count: {nonempty_cluster_count}")
print(f"centers shape: {result['centers'].shape}")
print("best_fitness <= final_population_best_fitness: True")
print("len(np.unique(labels)) == 6: True")
