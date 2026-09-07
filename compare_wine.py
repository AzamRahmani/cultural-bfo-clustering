import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_standardized_dataset
from bf_reusable import run_bf
from cbf_reusable import run_cbf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


X, y, X_scaled, feature_names, class_names = (
    load_standardized_dataset("wine")
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_fitness = float(kmeans.inertia_)

kmeans_silhouette = float(
    silhouette_score(
        X_scaled,
        kmeans_labels,
    )
)

bf_result = run_bf(
    X_scaled,
    num_clusters=3,
    random_seed=42,
)

cbf_result = run_cbf(
    X_scaled,
    num_clusters=3,
    random_seed=42,
)

results = [
    (
        "K-Means",
        kmeans_fitness,
        kmeans_silhouette,
    ),
    (
        "BF prototype",
        bf_result["best_fitness"],
        bf_result["silhouette"],
    ),
    (
        "CBF prototype",
        cbf_result["best_fitness"],
        cbf_result["silhouette"],
    ),
]

print("Wine Algorithm Comparison")
print("Samples: 178")
print("Features: 13")
print("Clusters: 3")
print("Random seed for BF and CBF: 42")
print()

for method, fitness, silhouette in results:
    print(f"Method: {method}")
    print(f"Fitness: {fitness:.4f}")
    print(f"Silhouette: {silhouette:.4f}")
    print()

lowest_fitness_result = min(
    results,
    key=lambda result: result[1],
)

highest_silhouette_result = max(
    results,
    key=lambda result: result[2],
)

print(f"Lowest fitness: {lowest_fitness_result[0]}")
print(
    f"Highest silhouette: "
    f"{highest_silhouette_result[0]}"
)
print()

assert X.shape == (178, 13)
assert y.shape == (178,)
assert X_scaled.shape == (178, 13)

assert kmeans_labels.shape == (178,)
assert bf_result["labels"].shape == (178,)
assert cbf_result["labels"].shape == (178,)

assert kmeans.cluster_centers_.shape == (3, 13)
assert bf_result["centers"].shape == (3, 13)
assert cbf_result["centers"].shape == (3, 13)

assert len(np.unique(kmeans_labels)) == 3
assert len(np.unique(bf_result["labels"])) == 3
assert len(np.unique(cbf_result["labels"])) == 3

assert all(
    np.isfinite(fitness)
    for _, fitness, _ in results
)

assert all(
    fitness > 0
    for _, fitness, _ in results
)

assert all(
    -1 <= silhouette <= 1
    for _, _, silhouette in results
)

assert abs(
    kmeans_fitness - 1277.9285
) < 0.001

assert abs(
    bf_result["best_fitness"] - 3366.1483
) < 0.001

assert abs(
    cbf_result["best_fitness"] - 2408.9080
) < 0.001

assert (
    cbf_result["best_fitness"]
    <= cbf_result["final_population_best_fitness"]
    + 1e-12
)

assert lowest_fitness_result[0] == "K-Means"
assert highest_silhouette_result[0] == "K-Means"
assert cbf_result["best_fitness"] < bf_result["best_fitness"]
assert cbf_result["silhouette"] > bf_result["silhouette"]

print("Important notes:")
print("- Lower fitness is better.")
print("- Higher silhouette is generally better.")
print(
    "- All methods use the same standardized Wine data "
    "and three clusters."
)
print("- BF and CBF use random seed 42.")
print(
    "- CBF fitness is the best archived fitness discovered "
    "during its run."
)
print("- This is a descriptive single-seed comparison.")
print("- The complete thesis method has not been reproduced.")
print()

method_names = [
    result[0]
    for result in results
]

fitness_values = [
    result[1]
    for result in results
]

silhouette_values = [
    result[2]
    for result in results
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
)

colors = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
]

fitness_bars = axes[0].bar(
    method_names,
    fitness_values,
    color=colors,
)

axes[0].set_title(
    "Wine Clustering Fitness (Lower Is Better)"
)

axes[0].set_xlabel("Method")

axes[0].set_ylabel(
    "Sum of Squared Distances"
)

axes[0].bar_label(
    fitness_bars,
    fmt="%.4f",
    padding=3,
)

silhouette_bars = axes[1].bar(
    method_names,
    silhouette_values,
    color=colors,
)

axes[1].set_title(
    "Wine Silhouette Score (Higher Is Better)"
)

axes[1].set_xlabel("Method")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_ylim(0, 0.35)

axes[1].bar_label(
    silhouette_bars,
    fmt="%.4f",
    padding=3,
)

fig.suptitle(
    "K-Means, BF, and CBF on Standardized Wine Data"
)

fig.text(
    0.5,
    0.01,
    "Single-seed descriptive comparison. "
    "BF and CBF remain exploratory prototypes.",
    ha="center",
    fontsize=9,
)

plt.tight_layout(
    rect=[0, 0.05, 1, 0.95]
)

output_file = "wine_algorithm_comparison.png"

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"Chart saved to: {output_file}")
print("Wine comparison checks passed.")
