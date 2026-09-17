import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bf_reusable import run_bf
from cbf_reusable import run_cbf
from data_loader import load_standardized_dataset


X, _, X_scaled, _, _ = load_standardized_dataset("glass")
seeds = [0, 1, 2, 3, 4]
num_clusters = 6

bf_results = []
cbf_results = []

for seed in seeds:
    bf_result = run_bf(
        X_scaled,
        num_clusters=num_clusters,
        random_seed=seed,
    )
    cbf_result = run_cbf(
        X_scaled,
        num_clusters=num_clusters,
        random_seed=seed,
    )

    bf_nonempty_cluster_count = len(
        np.unique(bf_result["labels"])
    )
    cbf_nonempty_cluster_count = len(
        np.unique(cbf_result["labels"])
    )

    assert len(np.unique(bf_result["labels"])) == 6
    assert len(np.unique(cbf_result["labels"])) == 6

    bf_results.append(
        {
            "seed": seed,
            "fitness": bf_result["best_fitness"],
            "silhouette": bf_result["silhouette"],
            "empty_cluster_rejections": (
                bf_result["empty_cluster_rejections"]
            ),
            "nonempty_cluster_count": (
                bf_nonempty_cluster_count
            ),
        }
    )
    cbf_results.append(
        {
            "seed": seed,
            "fitness": cbf_result["best_fitness"],
            "silhouette": cbf_result["silhouette"],
            "empty_cluster_rejections": (
                cbf_result["empty_cluster_rejections"]
            ),
            "nonempty_cluster_count": (
                cbf_nonempty_cluster_count
            ),
        }
    )


def metric_values(results, metric):
    return np.array([result[metric] for result in results])


bf_fitness_values = metric_values(bf_results, "fitness")
bf_silhouette_values = metric_values(
    bf_results,
    "silhouette",
)
cbf_fitness_values = metric_values(cbf_results, "fitness")
cbf_silhouette_values = metric_values(
    cbf_results,
    "silhouette",
)

print(f"X.shape: {X.shape}")
print("BF seed-by-seed results")
for result in bf_results:
    print(
        f"Seed {result['seed']}: "
        f"fitness={result['fitness']:.4f}, "
        f"silhouette={result['silhouette']:.4f}, "
        f"empty_cluster_rejections="
        f"{result['empty_cluster_rejections']}, "
        f"nonempty_clusters="
        f"{result['nonempty_cluster_count']}"
    )

print()
print("CBF seed-by-seed results")
for result in cbf_results:
    print(
        f"Seed {result['seed']}: "
        f"fitness={result['fitness']:.4f}, "
        f"silhouette={result['silhouette']:.4f}, "
        f"empty_cluster_rejections="
        f"{result['empty_cluster_rejections']}, "
        f"nonempty_clusters="
        f"{result['nonempty_cluster_count']}"
    )

print()
print("BF summary")
print(f"Mean fitness: {np.mean(bf_fitness_values):.4f}")
print(
    "Fitness standard deviation: "
    f"{np.std(bf_fitness_values):.4f}"
)
print(
    f"Mean silhouette: {np.mean(bf_silhouette_values):.4f}"
)
print(
    "Silhouette standard deviation: "
    f"{np.std(bf_silhouette_values):.4f}"
)

print()
print("CBF summary")
print(
    f"Mean fitness: {np.mean(cbf_fitness_values):.4f}"
)
print(
    "Fitness standard deviation: "
    f"{np.std(cbf_fitness_values):.4f}"
)
print(
    f"Mean silhouette: {np.mean(cbf_silhouette_values):.4f}"
)
print(
    "Silhouette standard deviation: "
    f"{np.std(cbf_silhouette_values):.4f}"
)

print()
print("BF rejection counts:", [
    result["empty_cluster_rejections"]
    for result in bf_results
])
print("CBF rejection counts:", [
    result["empty_cluster_rejections"]
    for result in cbf_results
])

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
)

axes[0].plot(
    seeds,
    bf_fitness_values,
    marker="o",
    label="BF",
)
axes[0].plot(
    seeds,
    cbf_fitness_values,
    marker="o",
    label="CBF",
)
axes[0].set_title("Glass Fitness Across Seeds (Lower Is Better)")
axes[0].set_xlabel("Random Seed")
axes[0].set_ylabel("Fitness")
axes[0].set_xticks(seeds)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    seeds,
    bf_silhouette_values,
    marker="o",
    label="BF",
)
axes[1].plot(
    seeds,
    cbf_silhouette_values,
    marker="o",
    label="CBF",
)
axes[1].set_title("Glass Silhouette Across Seeds (Higher Is Better)")
axes[1].set_xlabel("Random Seed")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(seeds)
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("BF and CBF Results Across Five Glass Experiments")
plt.tight_layout()
plt.savefig(
    "glass_seed_evaluation.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print()
print("Chart saved to: glass_seed_evaluation.png")
