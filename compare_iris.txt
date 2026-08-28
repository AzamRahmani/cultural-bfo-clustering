import io
from contextlib import redirect_stdout
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


with redirect_stdout(io.StringIO()):
    import iris_baseline
    import bf_iris
    import cbf_iris


results = [
    (
        "K-Means",
        float(iris_baseline.inertia),
        float(iris_baseline.silhouette),
    ),
    (
        "BF prototype",
        float(bf_iris.best_fitness),
        float(bf_iris.silhouette),
    ),
    (
        "CBF prototype",
        float(cbf_iris.best_fitness),
        float(cbf_iris.silhouette),
    ),
]

assert len(results) == 3
assert all(fitness > 0 for _, fitness, _ in results)
assert all(-1 <= score <= 1 for _, _, score in results)

print("Iris Clustering Comparison")
for method, fitness, silhouette in results:
    print(f"Method: {method}")
    print(f"Fitness: {fitness:.4f}")
    print(f"Silhouette: {silhouette:.4f}")

best_fitness_result = min(
    results,
    key=lambda result: result[1],
)
best_silhouette_result = max(
    results,
    key=lambda result: result[2],
)

print(f"Lowest fitness: {best_fitness_result[0]}")
print(f"Highest silhouette: {best_silhouette_result[0]}")

print()
print("Important notes:")
print("- Lower fitness is better.")
print("- Higher silhouette is generally better.")
print("- All methods use standardized Iris data and three clusters.")
print("- BF and CBF use one fixed random seed.")
print("- This comparison is descriptive, not statistical.")
print("- The CBF prototype is not the complete thesis implementation.")

method_names = [result[0] for result in results]
fitness_values = [result[1] for result in results]
silhouette_values = [result[2] for result in results]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 5),
)

fitness_bars = axes[0].bar(
    method_names,
    fitness_values,
    color=["#4C78A8", "#F58518", "#54A24B"],
)

axes[0].set_title("Iris Clustering Fitness (Lower Is Better)")
axes[0].set_ylabel("Sum of Squared Distances")
axes[0].set_xlabel("Method")
axes[0].bar_label(fitness_bars, fmt="%.4f", padding=3)

silhouette_bars = axes[1].bar(
    method_names,
    silhouette_values,
    color=["#4C78A8", "#F58518", "#54A24B"],
)

axes[1].set_title("Iris Silhouette Score (Higher Is Better)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xlabel("Method")
axes[1].set_ylim(0, 0.55)
axes[1].bar_label(silhouette_bars, fmt="%.4f", padding=3)

fig.suptitle(
    "K-Means, BF, and CBF Comparison on Standardized Iris Data"
)

plt.tight_layout()

output_file = "iris_comparison.png"

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print("Chart saved to: iris_comparison.png")