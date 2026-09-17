import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bf_reusable import run_bf
from cbf_reusable import run_cbf
from data_loader import load_standardized_dataset


X, y, X_scaled, _, _ = load_standardized_dataset("cancer")
seeds = [0, 1, 2, 3, 4]
num_clusters = 2

assert X_scaled.shape == (683, 9)
assert len(np.unique(y)) == 2

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

    bf_labels = bf_result["labels"]
    bf_centers = bf_result["centers"]
    bf_nonempty_cluster_count = len(np.unique(bf_labels))

    cbf_labels = cbf_result["labels"]
    cbf_centers = cbf_result["centers"]
    cbf_nonempty_cluster_count = len(np.unique(cbf_labels))
    cbf_centers_match_belief = np.array_equal(
        cbf_centers,
        cbf_result["belief_best_bacterium"],
    )

    assert bf_labels.shape == (683,)
    assert bf_centers.shape == (2, 9)
    assert bf_nonempty_cluster_count == 2
    assert np.isfinite(bf_result["best_fitness"])
    assert np.isfinite(bf_result["silhouette"])

    assert cbf_labels.shape == (683,)
    assert cbf_centers.shape == (2, 9)
    assert cbf_nonempty_cluster_count == 2
    assert np.isfinite(cbf_result["best_fitness"])
    assert np.isfinite(
        cbf_result["final_population_best_fitness"]
    )
    assert np.isfinite(cbf_result["silhouette"])
    assert cbf_result["best_fitness"] <= cbf_result[
        "final_population_best_fitness"
    ]
    assert cbf_centers_match_belief

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
            "final_population_best_fitness": (
                cbf_result["final_population_best_fitness"]
            ),
            "silhouette": cbf_result["silhouette"],
            "empty_cluster_rejections": (
                cbf_result["empty_cluster_rejections"]
            ),
            "nonempty_cluster_count": (
                cbf_nonempty_cluster_count
            ),
            "centers_match_belief": cbf_centers_match_belief,
        }
    )


def metric_values(results, metric):
    return np.array([result[metric] for result in results])


bf_fitness_values = metric_values(bf_results, "fitness")
bf_silhouette_values = metric_values(
    bf_results,
    "silhouette",
)
bf_rejection_values = metric_values(
    bf_results,
    "empty_cluster_rejections",
)
cbf_fitness_values = metric_values(cbf_results, "fitness")
cbf_silhouette_values = metric_values(
    cbf_results,
    "silhouette",
)
cbf_rejection_values = metric_values(
    cbf_results,
    "empty_cluster_rejections",
)

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
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
        f"final_population_best_fitness="
        f"{result['final_population_best_fitness']:.4f}, "
        f"silhouette={result['silhouette']:.4f}, "
        f"empty_cluster_rejections="
        f"{result['empty_cluster_rejections']}, "
        f"nonempty_clusters="
        f"{result['nonempty_cluster_count']}, "
        f"centers_match_belief="
        f"{result['centers_match_belief']}"
    )

print()
print("BF summary")
print(f"Mean fitness: {np.mean(bf_fitness_values):.4f}")
print(
    "Fitness standard deviation: "
    f"{np.std(bf_fitness_values):.4f}"
)
print(f"Mean silhouette: {np.mean(bf_silhouette_values):.4f}")
print(
    "Silhouette standard deviation: "
    f"{np.std(bf_silhouette_values):.4f}"
)
print(
    "Mean empty-cluster rejections: "
    f"{np.mean(bf_rejection_values):.4f}"
)
print(
    "Empty-cluster rejection standard deviation: "
    f"{np.std(bf_rejection_values):.4f}"
)

print()
print("CBF summary")
print(f"Mean fitness: {np.mean(cbf_fitness_values):.4f}")
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
print(
    "Mean empty-cluster rejections: "
    f"{np.mean(cbf_rejection_values):.4f}"
)
print(
    "Empty-cluster rejection standard deviation: "
    f"{np.std(cbf_rejection_values):.4f}"
)

print()
print(
    "BF empty-cluster rejection counts:",
    [result["empty_cluster_rejections"] for result in bf_results],
)
print(
    "CBF empty-cluster rejection counts:",
    [result["empty_cluster_rejections"] for result in cbf_results],
)
print(
    "BF nonempty cluster counts:",
    [result["nonempty_cluster_count"] for result in bf_results],
)
print(
    "CBF nonempty cluster counts:",
    [result["nonempty_cluster_count"] for result in cbf_results],
)

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
axes[0].set_title("Cancer Fitness Across Seeds (Lower Is Better)")
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
axes[1].set_title("Cancer Silhouette Across Seeds (Higher Is Better)")
axes[1].set_xlabel("Random Seed")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(seeds)
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("BF and CBF Results Across Five Cancer Experiments")
plt.tight_layout()
plt.savefig(
    "cancer_seed_evaluation.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print()
print("Chart saved to: cancer_seed_evaluation.png")
