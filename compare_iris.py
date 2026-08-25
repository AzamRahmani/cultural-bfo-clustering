import io
from contextlib import redirect_stdout


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