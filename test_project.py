import io
import numpy as np
from contextlib import redirect_stdout
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    glass_X, glass_y, glass_scaled, glass_features, glass_classes = (
        load_standardized_dataset("glass")
    )
    cancer_X, cancer_y, cancer_scaled, cancer_features, cancer_classes = (
        load_standardized_dataset("cancer")
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
    glass_kmeans = KMeans(
        n_clusters=6,
        random_state=42,
        n_init=10,
    )
    glass_kmeans_labels = glass_kmeans.fit_predict(glass_scaled)
    glass_bf = run_bf(
        glass_scaled,
        num_clusters=6,
        random_seed=42,
    )
    glass_cbf = run_cbf(
        glass_scaled,
        num_clusters=6,
        random_seed=42,
    )
    cancer_kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10,
    )
    cancer_kmeans_labels = cancer_kmeans.fit_predict(cancer_scaled)
    cancer_bf = run_bf(
        cancer_scaled,
        num_clusters=2,
        random_seed=42,
    )
    cancer_cbf = run_cbf(
        cancer_scaled,
        num_clusters=2,
        random_seed=42,
    )


def manual_inertia(X_scaled, labels, centers):
    return sum(
        ((sample - centers[labels[index]]) ** 2).sum()
        for index, sample in enumerate(X_scaled)
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

# Glass structural checks
assert glass_X.shape == (214, 9)
assert glass_y.shape == (214,)
assert glass_scaled.shape == (214, 9)
assert len(glass_features) == 9
assert len(glass_classes) == 6
assert len(np.unique(glass_y)) == 6

# Glass K-Means checks
assert np.isclose(
    glass_kmeans.inertia_,
    766.598324833444,
    rtol=0,
    atol=1e-9,
)
assert np.isclose(
    silhouette_score(glass_scaled, glass_kmeans_labels),
    0.324749440358,
    rtol=0,
    atol=1e-9,
)
assert glass_kmeans_labels.shape == (214,)
assert glass_kmeans.cluster_centers_.shape == (6, 9)
assert len(np.unique(glass_kmeans_labels)) == 6
assert np.isclose(
    manual_inertia(
        glass_scaled,
        glass_kmeans_labels,
        glass_kmeans.cluster_centers_,
    ),
    glass_kmeans.inertia_,
    rtol=1e-10,
    atol=1e-10,
)

# Glass reusable BF checks
assert np.isclose(
    glass_bf["best_fitness"],
    5593.8385,
    rtol=0,
    atol=0.001,
)
assert np.isclose(
    glass_bf["silhouette"],
    0.2135,
    rtol=0,
    atol=0.001,
)
assert glass_bf["empty_cluster_rejections"] == 534
assert glass_bf["labels"].shape == (214,)
assert glass_bf["centers"].shape == (6, 9)
assert len(np.unique(glass_bf["labels"])) == 6
assert np.all(
    np.isfinite(
        np.array(
            [
                glass_bf["best_fitness"],
                glass_bf["silhouette"],
                glass_bf["empty_cluster_rejections"],
            ],
            dtype=float,
        )
    )
)

# Glass reusable CBF checks
assert np.isclose(
    glass_cbf["best_fitness"],
    4335.7315,
    rtol=0,
    atol=0.001,
)
assert np.isclose(
    glass_cbf["final_population_best_fitness"],
    4335.7315,
    rtol=0,
    atol=0.001,
)
assert np.isclose(
    glass_cbf["silhouette"],
    0.2922,
    rtol=0,
    atol=0.001,
)
assert glass_cbf["empty_cluster_rejections"] == 615
assert glass_cbf["labels"].shape == (214,)
assert glass_cbf["centers"].shape == (6, 9)
assert len(np.unique(glass_cbf["labels"])) == 6
assert glass_cbf["best_fitness"] <= (
    glass_cbf["final_population_best_fitness"] + 1e-12
)
assert np.array_equal(
    glass_cbf["centers"],
    glass_cbf["belief_best_bacterium"],
)
assert np.all(
    np.isfinite(
        np.array(
            [
                glass_cbf["best_fitness"],
                glass_cbf["final_population_best_fitness"],
                glass_cbf["silhouette"],
                glass_cbf["empty_cluster_rejections"],
            ],
            dtype=float,
        )
    )
)

# Cancer structural checks
assert cancer_X.shape == (683, 9)
assert cancer_y.shape == (683,)
assert cancer_scaled.shape == (683, 9)
assert len(cancer_features) == 9
assert len(cancer_classes) == 2
assert len(np.unique(cancer_y)) == 2

# Cancer K-Means checks
assert np.isclose(
    cancer_kmeans.inertia_,
    2728.4358,
    rtol=0,
    atol=0.001,
)
assert np.isclose(
    silhouette_score(cancer_scaled, cancer_kmeans_labels),
    0.5734,
    rtol=0,
    atol=0.001,
)
assert cancer_kmeans_labels.shape == (683,)
assert cancer_kmeans.cluster_centers_.shape == (2, 9)
assert len(np.unique(cancer_kmeans_labels)) == 2
assert np.isclose(
    manual_inertia(
        cancer_scaled,
        cancer_kmeans_labels,
        cancer_kmeans.cluster_centers_,
    ),
    cancer_kmeans.inertia_,
    rtol=1e-10,
    atol=1e-10,
)

# Cancer reusable BF checks
assert np.isclose(
    cancer_bf["best_fitness"],
    4763.2108,
    rtol=0,
    atol=0.001,
)
assert np.isclose(
    cancer_bf["silhouette"],
    0.5471,
    rtol=0,
    atol=0.001,
)
assert cancer_bf["empty_cluster_rejections"] == 0
assert cancer_bf["labels"].shape == (683,)
assert cancer_bf["centers"].shape == (2, 9)
assert len(np.unique(cancer_bf["labels"])) == 2
assert np.all(
    np.isfinite(
        np.array(
            [
                cancer_bf["best_fitness"],
                cancer_bf["silhouette"],
                cancer_bf["empty_cluster_rejections"],
            ],
            dtype=float,
        )
    )
)

# Cancer reusable CBF checks
assert np.isclose(
    cancer_cbf["best_fitness"],
    4492.6286,
    rtol=0,
    atol=0.001,
)
assert np.isclose(
    cancer_cbf["final_population_best_fitness"],
    4492.6286,
    rtol=0,
    atol=0.001,
)
assert np.isclose(
    cancer_cbf["silhouette"],
    0.5649,
    rtol=0,
    atol=0.001,
)
assert cancer_cbf["empty_cluster_rejections"] == 0
assert cancer_cbf["labels"].shape == (683,)
assert cancer_cbf["centers"].shape == (2, 9)
assert len(np.unique(cancer_cbf["labels"])) == 2
assert cancer_cbf["best_fitness"] <= (
    cancer_cbf["final_population_best_fitness"] + 1e-12
)
assert np.array_equal(
    cancer_cbf["centers"],
    cancer_cbf["belief_best_bacterium"],
)
assert np.all(
    np.isfinite(
        np.array(
            [
                cancer_cbf["best_fitness"],
                cancer_cbf["final_population_best_fitness"],
                cancer_cbf["silhouette"],
                cancer_cbf["empty_cluster_rejections"],
            ],
            dtype=float,
        )
    )
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
