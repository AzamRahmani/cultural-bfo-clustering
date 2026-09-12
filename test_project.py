import io
import numpy as np
from contextlib import redirect_stdout


with redirect_stdout(io.StringIO()):
    import iris_baseline
    import bf_iris
    import cbf_iris
    from data_loader import load_standardized_dataset
    from bf_reusable import run_bf
    from cbf_reusable import run_cbf

    _, _, reusable_iris_scaled, _, _ = (
        load_standardized_dataset("iris")
    )
    _, _, reusable_wine_scaled, _, _ = (
        load_standardized_dataset("wine")
    )

    reusable_iris_bf = run_bf(
        reusable_iris_scaled,
        num_clusters=3,
        random_seed=42,
    )
    reusable_iris_cbf = run_cbf(
        reusable_iris_scaled,
        num_clusters=3,
        random_seed=42,
    )
    reusable_wine_bf = run_bf(
        reusable_wine_scaled,
        num_clusters=3,
        random_seed=42,
    )
    reusable_wine_cbf = run_cbf(
        reusable_wine_scaled,
        num_clusters=3,
        random_seed=42,
    )


assert iris_baseline.X.shape == (150, 4)
assert iris_baseline.y.shape == (150,)
assert iris_baseline.kmeans_labels.shape == (150,)
assert iris_baseline.kmeans.cluster_centers_.shape == (3, 4)
assert np.isfinite(iris_baseline.inertia)
assert iris_baseline.inertia > 0
assert -1 <= iris_baseline.silhouette <= 1
assert iris_baseline.difference < 1e-10

assert bf_iris.X_scaled.shape == (150, 4)
assert bf_iris.bacteria.shape == (20, 3, 4)
assert bf_iris.fitness_values.shape == (20,)
assert bf_iris.health_values.shape == (20,)
assert bf_iris.best_labels.shape == (150,)
assert np.isfinite(bf_iris.best_fitness)
assert bf_iris.best_fitness > 0
assert -1 <= bf_iris.silhouette <= 1
assert len(np.unique(bf_iris.best_labels)) == 3
assert bf_iris.health_accumulation_steps == 50
assert bf_iris.reproduction_events == 5
assert bf_iris.elimination_events == 5
assert len(bf_iris.bacteria) == 20

assert cbf_iris.X_scaled.shape == (150, 4)
assert cbf_iris.bacteria.shape == (20, 3, 4)
assert cbf_iris.fitness_values.shape == (20,)
assert cbf_iris.health_values.shape == (20,)
assert cbf_iris.best_labels.shape == (150,)
assert cbf_iris.belief_best_bacterium.shape == (3, 4)
assert cbf_iris.normative_lower.shape == (3, 4)
assert cbf_iris.normative_upper.shape == (3, 4)
assert np.all(cbf_iris.normative_lower <= cbf_iris.normative_upper)
assert np.isfinite(cbf_iris.best_fitness)
assert np.isfinite(cbf_iris.belief_best_fitness)
assert cbf_iris.belief_best_fitness <= cbf_iris.best_fitness + 1e-12
assert -1 <= cbf_iris.silhouette <= 1
assert len(np.unique(cbf_iris.best_labels)) == 3
assert cbf_iris.accepted_count == 7
assert cbf_iris.normative_updates == 50
assert cbf_iris.health_accumulation_steps == 50
assert cbf_iris.reproduction_events == 5
assert cbf_iris.elimination_events == 5
assert len(cbf_iris.bacteria) == 20

assert reusable_iris_bf["labels"].shape == (150,)
assert reusable_iris_bf["centers"].shape == (3, 4)
assert reusable_iris_bf["population"].shape == (20, 3, 4)
assert reusable_iris_bf["fitness_values"].shape == (20,)
assert reusable_iris_bf["health_values"].shape == (20,)
assert len(np.unique(reusable_iris_bf["labels"])) == 3
assert np.isfinite(reusable_iris_bf["best_fitness"])
assert reusable_iris_bf["best_fitness"] > 0
assert np.all(np.isfinite(reusable_iris_bf["fitness_values"]))
assert -1 <= reusable_iris_bf["silhouette"] <= 1
assert reusable_iris_bf["empty_cluster_rejections"] >= 0
assert reusable_iris_bf["health_accumulation_steps"] == 50
assert reusable_iris_bf["reproduction_events"] == 5
assert reusable_iris_bf["elimination_events"] == 5
assert len(reusable_iris_bf["population"]) == 20

assert reusable_iris_cbf["labels"].shape == (150,)
assert reusable_iris_cbf["centers"].shape == (3, 4)
assert reusable_iris_cbf["population"].shape == (20, 3, 4)
assert reusable_iris_cbf["belief_best_bacterium"].shape == (3, 4)
assert reusable_iris_cbf["normative_lower"].shape == (3, 4)
assert reusable_iris_cbf["normative_upper"].shape == (3, 4)
assert reusable_iris_cbf["fitness_values"].shape == (20,)
assert reusable_iris_cbf["health_values"].shape == (20,)
assert len(np.unique(reusable_iris_cbf["labels"])) == 3
assert np.isfinite(reusable_iris_cbf["best_fitness"])
assert reusable_iris_cbf["best_fitness"] > 0
assert np.isfinite(
    reusable_iris_cbf["final_population_best_fitness"]
)
assert np.all(np.isfinite(reusable_iris_cbf["fitness_values"]))
assert -1 <= reusable_iris_cbf["silhouette"] <= 1
assert reusable_iris_cbf["empty_cluster_rejections"] >= 0
assert reusable_iris_cbf["accepted_count"] == 7
assert reusable_iris_cbf["normative_updates"] == 50
assert reusable_iris_cbf["health_accumulation_steps"] == 50
assert reusable_iris_cbf["reproduction_events"] == 5
assert reusable_iris_cbf["elimination_events"] == 5
assert len(reusable_iris_cbf["population"]) == 20
assert np.all(
    reusable_iris_cbf["normative_lower"]
    <= reusable_iris_cbf["normative_upper"]
)
assert (
    reusable_iris_cbf["best_fitness"]
    <= reusable_iris_cbf["final_population_best_fitness"] + 1e-12
)

assert reusable_wine_bf["labels"].shape == (178,)
assert reusable_wine_bf["centers"].shape == (3, 13)
assert reusable_wine_bf["population"].shape == (20, 3, 13)
assert reusable_wine_bf["fitness_values"].shape == (20,)
assert reusable_wine_bf["health_values"].shape == (20,)
assert len(np.unique(reusable_wine_bf["labels"])) == 3
assert np.isfinite(reusable_wine_bf["best_fitness"])
assert reusable_wine_bf["best_fitness"] > 0
assert np.all(np.isfinite(reusable_wine_bf["fitness_values"]))
assert -1 <= reusable_wine_bf["silhouette"] <= 1
assert reusable_wine_bf["empty_cluster_rejections"] >= 0
assert reusable_wine_bf["health_accumulation_steps"] == 50
assert reusable_wine_bf["reproduction_events"] == 5
assert reusable_wine_bf["elimination_events"] == 5
assert len(reusable_wine_bf["population"]) == 20

assert reusable_wine_cbf["labels"].shape == (178,)
assert reusable_wine_cbf["centers"].shape == (3, 13)
assert reusable_wine_cbf["population"].shape == (20, 3, 13)
assert reusable_wine_cbf["belief_best_bacterium"].shape == (3, 13)
assert reusable_wine_cbf["normative_lower"].shape == (3, 13)
assert reusable_wine_cbf["normative_upper"].shape == (3, 13)
assert reusable_wine_cbf["fitness_values"].shape == (20,)
assert reusable_wine_cbf["health_values"].shape == (20,)
assert len(np.unique(reusable_wine_cbf["labels"])) == 3
assert np.isfinite(reusable_wine_cbf["best_fitness"])
assert reusable_wine_cbf["best_fitness"] > 0
assert np.isfinite(
    reusable_wine_cbf["final_population_best_fitness"]
)
assert np.all(np.isfinite(reusable_wine_cbf["fitness_values"]))
assert -1 <= reusable_wine_cbf["silhouette"] <= 1
assert reusable_wine_cbf["empty_cluster_rejections"] >= 0
assert reusable_wine_cbf["accepted_count"] == 7
assert reusable_wine_cbf["normative_updates"] == 50
assert reusable_wine_cbf["health_accumulation_steps"] == 50
assert reusable_wine_cbf["reproduction_events"] == 5
assert reusable_wine_cbf["elimination_events"] == 5
assert len(reusable_wine_cbf["population"]) == 20
assert np.all(
    reusable_wine_cbf["normative_lower"]
    <= reusable_wine_cbf["normative_upper"]
)
assert (
    reusable_wine_cbf["best_fitness"]
    <= reusable_wine_cbf["final_population_best_fitness"] + 1e-12
)

assert abs(reusable_iris_bf["best_fitness"] - 176.8929) < 0.001
assert abs(reusable_iris_cbf["best_fitness"] - 152.6087) < 0.001
assert abs(reusable_wine_bf["best_fitness"] - 3366.1483) < 0.001
assert abs(reusable_wine_cbf["best_fitness"] - 2316.0097) < 0.001

print("Reusable algorithm checks passed.")
print(
    "Iris reusable BF fitness: "
    f"{reusable_iris_bf['best_fitness']:.4f}"
)
print(
    "Iris reusable CBF fitness: "
    f"{reusable_iris_cbf['best_fitness']:.4f}"
)
print(
    "Wine reusable BF fitness: "
    f"{reusable_wine_bf['best_fitness']:.4f}"
)
print(
    "Wine reusable CBF fitness: "
    f"{reusable_wine_cbf['best_fitness']:.4f}"
)
print(
    "Iris BF empty-cluster rejections: "
    f"{reusable_iris_bf['empty_cluster_rejections']}"
)
print(
    "Iris CBF empty-cluster rejections: "
    f"{reusable_iris_cbf['empty_cluster_rejections']}"
)
print(
    "Wine BF empty-cluster rejections: "
    f"{reusable_wine_bf['empty_cluster_rejections']}"
)
print(
    "Wine CBF empty-cluster rejections: "
    f"{reusable_wine_cbf['empty_cluster_rejections']}"
)
print("All reusable results contain three nonempty clusters.")

print("Project checks passed.")
print(f"K-Means fitness: {iris_baseline.inertia:.4f}")
print(f"BF fitness: {bf_iris.best_fitness:.4f}")
print(f"CBF fitness: {cbf_iris.best_fitness:.4f}")
print("All methods produced three nonempty clusters.")
