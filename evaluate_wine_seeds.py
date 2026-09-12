import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_standardized_dataset
from bf_reusable import run_bf
from cbf_reusable import run_cbf


X, y, X_scaled, feature_names, class_names = (
    load_standardized_dataset("wine")
)

seeds = [0, 1, 2, 3, 4]

bf_fitness_values = []
bf_silhouette_values = []
cbf_fitness_values = []
cbf_silhouette_values = []

for seed in seeds:
    bf_result = run_bf(
        X_scaled,
        num_clusters=3,
        random_seed=seed,
    )

    cbf_result = run_cbf(
        X_scaled,
        num_clusters=3,
        random_seed=seed,
    )

    bf_fitness_values.append(
        bf_result["best_fitness"]
    )
    bf_silhouette_values.append(
        bf_result["silhouette"]
    )
    cbf_fitness_values.append(
        cbf_result["best_fitness"]
    )
    cbf_silhouette_values.append(
        cbf_result["silhouette"]
    )

    assert bf_result["labels"].shape == (178,)
    assert bf_result["centers"].shape == (3, 13)
    assert bf_result["population"].shape == (
        20,
        3,
        13,
    )
    assert len(np.unique(bf_result["labels"])) == 3
    assert np.isfinite(
        bf_result["best_fitness"]
    )
    assert bf_result["best_fitness"] > 0
    assert -1 <= bf_result["silhouette"] <= 1

    assert cbf_result["labels"].shape == (178,)
    assert cbf_result["centers"].shape == (3, 13)
    assert cbf_result["population"].shape == (
        20,
        3,
        13,
    )
    assert len(np.unique(cbf_result["labels"])) == 3
    assert np.isfinite(
        cbf_result["best_fitness"]
    )
    assert np.isfinite(
        cbf_result[
            "final_population_best_fitness"
        ]
    )
    assert cbf_result["best_fitness"] > 0
    assert -1 <= cbf_result["silhouette"] <= 1
    assert (
        cbf_result["best_fitness"]
        <= cbf_result[
            "final_population_best_fitness"
        ] + 1e-12
    )

    print(f"Seed: {seed}")
    print(
        f"BF fitness: "
        f"{bf_result['best_fitness']:.4f}"
    )
    print(
        f"BF silhouette: "
        f"{bf_result['silhouette']:.4f}"
    )
    print(
        f"CBF archived fitness: "
        f"{cbf_result['best_fitness']:.4f}"
    )
    print(
        f"CBF final-population fitness: "
        f"{cbf_result['final_population_best_fitness']:.4f}"
    )
    print(
        f"CBF silhouette: "
        f"{cbf_result['silhouette']:.4f}"
    )
    print()

bf_fitness_values = np.array(
    bf_fitness_values
)

bf_silhouette_values = np.array(
    bf_silhouette_values
)

cbf_fitness_values = np.array(
    cbf_fitness_values
)

cbf_silhouette_values = np.array(
    cbf_silhouette_values
)

bf_mean_fitness = np.mean(
    bf_fitness_values
)

bf_fitness_std = np.std(
    bf_fitness_values,
    ddof=0,
)

bf_mean_silhouette = np.mean(
    bf_silhouette_values
)

bf_silhouette_std = np.std(
    bf_silhouette_values,
    ddof=0,
)

cbf_mean_fitness = np.mean(
    cbf_fitness_values
)

cbf_fitness_std = np.std(
    cbf_fitness_values,
    ddof=0,
)

cbf_mean_silhouette = np.mean(
    cbf_silhouette_values
)

cbf_silhouette_std = np.std(
    cbf_silhouette_values,
    ddof=0,
)

bf_fitness_wins = int(
    np.sum(
        bf_fitness_values
        < cbf_fitness_values
    )
)

cbf_fitness_wins = int(
    np.sum(
        cbf_fitness_values
        < bf_fitness_values
    )
)

bf_silhouette_wins = int(
    np.sum(
        bf_silhouette_values
        > cbf_silhouette_values
    )
)

cbf_silhouette_wins = int(
    np.sum(
        cbf_silhouette_values
        > bf_silhouette_values
    )
)

assert X.shape == (178, 13)
assert y.shape == (178,)
assert X_scaled.shape == (178, 13)

assert len(bf_fitness_values) == 5
assert len(cbf_fitness_values) == 5
assert len(bf_silhouette_values) == 5
assert len(cbf_silhouette_values) == 5

assert np.all(
    np.isfinite(bf_fitness_values)
)

assert np.all(
    np.isfinite(cbf_fitness_values)
)

assert np.all(
    np.isfinite(bf_silhouette_values)
)

assert np.all(
    np.isfinite(cbf_silhouette_values)
)

assert np.all(bf_fitness_values > 0)
assert np.all(cbf_fitness_values > 0)

assert np.all(
    (-1 <= bf_silhouette_values)
    & (bf_silhouette_values <= 1)
)

assert np.all(
    (-1 <= cbf_silhouette_values)
    & (cbf_silhouette_values <= 1)
)

assert (
    bf_fitness_wins + cbf_fitness_wins
    <= 5
)

assert (
    bf_silhouette_wins
    + cbf_silhouette_wins
    <= 5
)

print("Wine Five-Seed Evaluation")
print()
print(
    f"BF mean fitness: "
    f"{bf_mean_fitness:.4f}"
)
print(
    f"BF fitness standard deviation: "
    f"{bf_fitness_std:.4f}"
)
print(
    f"BF mean silhouette: "
    f"{bf_mean_silhouette:.4f}"
)
print(
    f"BF silhouette standard deviation: "
    f"{bf_silhouette_std:.4f}"
)
print()
print(
    f"CBF mean fitness: "
    f"{cbf_mean_fitness:.4f}"
)
print(
    f"CBF fitness standard deviation: "
    f"{cbf_fitness_std:.4f}"
)
print(
    f"CBF mean silhouette: "
    f"{cbf_mean_silhouette:.4f}"
)
print(
    f"CBF silhouette standard deviation: "
    f"{cbf_silhouette_std:.4f}"
)
print()
print(
    f"BF lower-fitness wins: "
    f"{bf_fitness_wins} of 5"
)
print(
    f"CBF lower-fitness wins: "
    f"{cbf_fitness_wins} of 5"
)
print(
    f"BF higher-silhouette wins: "
    f"{bf_silhouette_wins} of 5"
)
print(
    f"CBF higher-silhouette wins: "
    f"{cbf_silhouette_wins} of 5"
)

print()
print("Important notes:")
print("- Lower fitness is better.")
print("- Higher silhouette is generally better.")
print(
    "- Each seed produces a different "
    "random search sequence."
)
print(
    "- CBF fitness refers to its "
    "archived best solution."
)
print(
    "- Five seeds provide an initial "
    "stability check."
)
print(
    "- Five seeds are insufficient "
    "for statistical significance."
)
print(
    "- The complete thesis CBF method "
    "has not been reproduced."
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

axes[0].set_title(
    "Wine Fitness Across Seeds (Lower Is Better)"
)

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

axes[1].set_title(
    "Wine Silhouette Across Seeds (Higher Is Better)"
)

axes[1].set_xlabel("Random Seed")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(seeds)
axes[1].set_ylim(0, 0.4)
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle(
    "Reusable BF and CBF Across Five Wine Experiments"
)

fig.text(
    0.5,
    0.01,
    "Each point is one complete run. "
    "Five seeds are not statistically conclusive.",
    ha="center",
    fontsize=9,
)

plt.tight_layout(
    rect=[0, 0.05, 1, 0.95]
)

output_file = "wine_seed_evaluation.png"

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    "Chart saved to: "
    "wine_seed_evaluation.png"
)

print(
    "Wine seed-evaluation checks passed."
)
