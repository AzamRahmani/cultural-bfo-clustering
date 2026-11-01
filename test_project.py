import csv
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from clustering_metrics import (
    within_cluster_sum_euclidean_distance,
    within_cluster_sum_squared_distance,
)
from clustering_evaluation import clustering_error_rate
from bacterial_interaction import bacterial_interaction
from experiment_results import (
    EXPERIMENT_RESULT_COLUMNS,
    read_experiment_results,
    write_experiment_results,
)


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


def assert_runtime_and_evaluation_count(result):
    assert "runtime_seconds" in result
    assert isinstance(result["runtime_seconds"], float)
    assert np.isfinite(result["runtime_seconds"])
    assert result["runtime_seconds"] >= 0
    assert "fitness_evaluation_count" in result
    assert isinstance(result["fitness_evaluation_count"], int)
    assert result["fitness_evaluation_count"] > 0


def assert_same_seed_results(first, second, event_counter_names):
    assert (
        first["fitness_evaluation_count"]
        == second["fitness_evaluation_count"]
    )
    assert np.isclose(
        first["best_fitness"],
        second["best_fitness"],
        rtol=0,
        atol=1e-12,
    )
    assert np.array_equal(first["labels"], second["labels"])
    assert np.array_equal(first["centers"], second["centers"])
    for counter_name in event_counter_names:
        assert first[counter_name] == second[counter_name]


for dataset_name, algorithm_name, reusable_result in (
    ("Iris", "BF", reusable_iris_bf),
    ("Iris", "CBF", reusable_iris_cbf),
    ("Wine", "BF", reusable_wine_bf),
    ("Wine", "CBF", reusable_wine_cbf),
    ("Glass", "BF", glass_bf),
    ("Glass", "CBF", glass_cbf),
    ("Cancer", "BF", cancer_bf),
    ("Cancer", "CBF", cancer_cbf),
):
    assert_runtime_and_evaluation_count(reusable_result)
    print(
        f"{dataset_name} {algorithm_name}: "
        "fitness_evaluation_count="
        f"{reusable_result['fitness_evaluation_count']}, "
        "runtime_seconds (environment-dependent)="
        f"{reusable_result['runtime_seconds']:.9f}"
    )


same_seed_parameters = {
    "num_clusters": 3,
    "random_seed": 1729,
    "num_iterations": 3,
    "reproduction_interval": 1,
    "elimination_interval": 1,
}
same_seed_bf_first = run_bf(
    reusable_iris_scaled,
    **same_seed_parameters,
)
same_seed_bf_second = run_bf(
    reusable_iris_scaled,
    **same_seed_parameters,
)
same_seed_cbf_first = run_cbf(
    reusable_iris_scaled,
    **same_seed_parameters,
)
same_seed_cbf_second = run_cbf(
    reusable_iris_scaled,
    **same_seed_parameters,
)

assert_same_seed_results(
    same_seed_bf_first,
    same_seed_bf_second,
    (
        "accepted_movements",
        "health_accumulation_steps",
        "reproduction_events",
        "elimination_events",
        "dispersed_bacteria_count",
        "empty_cluster_rejections",
    ),
)
assert_same_seed_results(
    same_seed_cbf_first,
    same_seed_cbf_second,
    (
        "accepted_movements",
        "cultural_movements",
        "belief_updates",
        "normative_updates",
        "normative_bounds_changes",
        "normative_clipped_candidates",
        "health_accumulation_steps",
        "reproduction_events",
        "elimination_events",
        "dispersed_bacteria_count",
        "empty_cluster_rejections",
    ),
)


metric_X = np.array([[0.0, 0.0], [3.0, 4.0]])
metric_labels = np.array([0, 0])
metric_centers = np.array([[0.0, 0.0]])
assert within_cluster_sum_euclidean_distance(
    metric_X, metric_labels, metric_centers
) == 5.0
assert within_cluster_sum_squared_distance(
    metric_X, metric_labels, metric_centers
) == 25.0

multi_cluster_X = np.array([[1.0, 0.0], [9.0, 0.0], [3.0, 4.0]])
multi_cluster_labels = np.array([0, 1, 0])
multi_cluster_centers = np.array([[0.0, 0.0], [10.0, 0.0]])
assert within_cluster_sum_euclidean_distance(
    multi_cluster_X, multi_cluster_labels, multi_cluster_centers
) == 7.0
assert within_cluster_sum_squared_distance(
    multi_cluster_X, multi_cluster_labels, multi_cluster_centers
) == 27.0

bacterial_candidate = np.array([0.0, 0.0])
bacterial_population = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
)
bacterial_parameters = (2.0, 0.5, 3.0, 1.0)
zero_distance_interaction = -2.0 + 3.0
unit_distance_interaction = (
    -2.0 * np.exp(-0.5)
    + 3.0 * np.exp(-1.0)
)
assert np.isclose(
    bacterial_interaction(
        bacterial_candidate,
        bacterial_population,
        *bacterial_parameters,
    ),
    2 * zero_distance_interaction + unit_distance_interaction,
)
assert np.isclose(
    bacterial_interaction(
        bacterial_candidate,
        bacterial_population,
        *bacterial_parameters,
        include_self=False,
        self_index=0,
    ),
    zero_distance_interaction + unit_distance_interaction,
)


def assert_bacterial_interaction_value_error(
    candidate,
    population,
    expected_message,
    include_self=True,
    self_index=None,
):
    try:
        bacterial_interaction(
            candidate,
            population,
            *bacterial_parameters,
            include_self=include_self,
            self_index=self_index,
        )
    except ValueError as error:
        assert expected_message in str(error)
    else:
        raise AssertionError(
            "Expected bacterial interaction input validation to fail."
        )


assert_bacterial_interaction_value_error(
    [[0.0, 0.0]],
    bacterial_population,
    "candidate must have shape",
)
assert_bacterial_interaction_value_error(
    bacterial_candidate,
    [0.0, 0.0],
    "population must have shape",
)
assert_bacterial_interaction_value_error(
    bacterial_candidate,
    [[0.0, 0.0, 0.0]],
    "same number of dimensions",
)
assert_bacterial_interaction_value_error(
    bacterial_candidate,
    bacterial_population,
    "self_index is required",
    include_self=False,
)
assert_bacterial_interaction_value_error(
    bacterial_candidate,
    bacterial_population,
    "identify a row",
    include_self=False,
    self_index=3,
)


def assert_metric_value_error(function, X, labels, centers, expected_message):
    try:
        function(X, labels, centers)
    except ValueError as error:
        assert expected_message in str(error)
    else:
        raise AssertionError("Expected metric input validation to fail.")


assert_metric_value_error(
    within_cluster_sum_squared_distance,
    [[0.0, 0.0], [1.0, 1.0]],
    [0],
    [[0.0, 0.0]],
    "same number of samples",
)
assert_metric_value_error(
    within_cluster_sum_squared_distance,
    [[0.0, 0.0]],
    [0],
    [[0.0, 0.0, 0.0]],
    "same number of features",
)
assert_metric_value_error(
    within_cluster_sum_squared_distance,
    [[0.0, 0.0]],
    [1],
    [[0.0, 0.0]],
    "valid center indices",
)
assert_metric_value_error(
    within_cluster_sum_euclidean_distance,
    [[np.inf, 0.0]],
    [0],
    [[0.0, 0.0]],
    "finite real numeric values",
)
assert_metric_value_error(
    within_cluster_sum_squared_distance,
    [[0.0, 0.0]],
    [0.0],
    [[0.0, 0.0]],
    "integer cluster identifiers",
)


def assert_clustering_evaluation_value_error(
    y_true,
    cluster_labels,
    expected_message,
):
    try:
        clustering_error_rate(y_true, cluster_labels)
    except ValueError as error:
        assert expected_message in str(error)
    else:
        raise AssertionError(
            "Expected clustering evaluation input validation to fail."
        )


swapped_evaluation = clustering_error_rate(
    [0, 0, 1, 1],
    [1, 1, 0, 0],
)
assert swapped_evaluation["mapping"] == {1: 0, 0: 1}
assert swapped_evaluation["correct_count"] == 4
assert swapped_evaluation["incorrect_count"] == 0
assert swapped_evaluation["accuracy_percent"] == 100.0
assert swapped_evaluation["error_rate_percent"] == 0.0
assert np.array_equal(
    swapped_evaluation["contingency_matrix"],
    np.array([[0, 2], [2, 0]]),
)

partial_evaluation = clustering_error_rate(
    [0, 0, 1, 1],
    [1, 0, 0, 0],
)
assert partial_evaluation["correct_count"] == 3
assert partial_evaluation["incorrect_count"] == 1
assert partial_evaluation["accuracy_percent"] == 75.0
assert partial_evaluation["error_rate_percent"] == 25.0

arbitrary_identifier_evaluation = clustering_error_rate(
    [2, 2, 4, 4],
    [20, 20, 10, 10],
)
assert arbitrary_identifier_evaluation["mapping"] == {
    20: 2,
    10: 4,
}
assert arbitrary_identifier_evaluation["accuracy_percent"] == 100.0

assert_clustering_evaluation_value_error(
    [0, 1],
    [0],
    "same number of samples",
)
assert_clustering_evaluation_value_error(
    [[0, 1]],
    [0, 1],
    "one-dimensional array",
)
assert_clustering_evaluation_value_error(
    [0, 1],
    [[0, 1]],
    "one-dimensional array",
)
assert_clustering_evaluation_value_error(
    [],
    [],
    "At least one sample",
)
assert_clustering_evaluation_value_error(
    [0, 0, 1],
    [10, 20, 30],
    "number of predicted clusters",
)
assert_clustering_evaluation_value_error(
    [0.0, np.nan, 1.0],
    [10, 20, 20],
    "missing or non-finite",
)
assert_clustering_evaluation_value_error(
    [0, 1, 1],
    [10.0, np.inf, 20.0],
    "missing or non-finite",
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


def make_experiment_result_row(seed=7, parameter_configuration=None):
    if parameter_configuration is None:
        parameter_configuration = {"step_size": 0.1, "num_clusters": 3}
    return {
        "dataset": "Iris",
        "algorithm": "BF",
        "seed": seed,
        "best_fitness": 12.5,
        "final_population_best_fitness": None,
        "silhouette": 0.72,
        "error_rate_percent": 4.0,
        "accuracy_percent": 96.0,
        "correct_count": 144,
        "incorrect_count": 6,
        "nonempty_cluster_count": 3,
        "empty_cluster_rejections": 2,
        "runtime_seconds": 1.25,
        "fitness_evaluation_count": 80,
        "accepted_movements": 11,
        "reproduction_events": 5,
        "elimination_events": 5,
        "dispersed_bacteria_count": 2,
        "run_status": "success",
        "failure_description": "",
        "parameter_configuration": parameter_configuration,
        "git_commit": "abc123",
        "python_version": "3.12.0",
        "numpy_version": "2.0.0",
        "sklearn_version": "1.5.0",
        "scipy_version": "1.14.0",
        "matplotlib_version": "3.9.0",
    }


def assert_experiment_results_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Expected experiment results validation to fail.")


with tempfile.TemporaryDirectory() as temporary_directory:
    temporary_root = Path(temporary_directory)
    output_path = temporary_root / "nested" / "results.csv"
    expected_experiment_columns = (
        "dataset",
        "algorithm",
        "seed",
        "best_fitness",
        "final_population_best_fitness",
        "silhouette",
        "error_rate_percent",
        "accuracy_percent",
        "correct_count",
        "incorrect_count",
        "nonempty_cluster_count",
        "empty_cluster_rejections",
        "runtime_seconds",
        "fitness_evaluation_count",
        "accepted_movements",
        "reproduction_events",
        "elimination_events",
        "dispersed_bacteria_count",
        "run_status",
        "failure_description",
        "parameter_configuration",
        "git_commit",
        "python_version",
        "numpy_version",
        "sklearn_version",
        "scipy_version",
        "matplotlib_version",
    )
    assert EXPERIMENT_RESULT_COLUMNS == expected_experiment_columns

    first_configuration = {
        "step_size": 0.1,
        "limits": {"upper": 5, "lower": -5},
    }
    second_configuration = {
        "limits": {"lower": -5, "upper": 5},
        "step_size": 0.1,
    }
    experiment_rows = [
        make_experiment_result_row(7, first_configuration),
        make_experiment_result_row(8, second_configuration),
    ]

    write_experiment_results(experiment_rows, output_path)
    assert output_path.parent.is_dir()
    with output_path.open("r", encoding="utf-8", newline="") as result_file:
        csv_reader = csv.reader(result_file)
        assert next(csv_reader) == list(expected_experiment_columns)
        serialized_configurations = [
            row[expected_experiment_columns.index("parameter_configuration")]
            for row in csv_reader
        ]
    assert serialized_configurations == [
        '{"limits":{"lower":-5,"upper":5},"step_size":0.1}',
        '{"limits":{"lower":-5,"upper":5},"step_size":0.1}',
    ]

    loaded_rows = read_experiment_results(output_path)
    assert len(loaded_rows) == len(experiment_rows)
    assert loaded_rows[0]["dataset"] == "Iris"
    assert loaded_rows[0]["seed"] == "7"
    assert loaded_rows[0]["parameter_configuration"] == first_configuration
    assert loaded_rows[0]["run_status"] == "success"
    assert loaded_rows[0]["failure_description"] == ""
    assert loaded_rows[0]["matplotlib_version"] == "3.9.0"
    assert loaded_rows[0]["final_population_best_fitness"] is None

    with output_path.open("r", encoding="utf-8", newline="") as result_file:
        csv_reader = csv.reader(result_file)
        next(csv_reader)
        first_csv_row = next(csv_reader)
    assert first_csv_row[
        expected_experiment_columns.index("final_population_best_fitness")
    ] == ""

    try:
        write_experiment_results(experiment_rows, output_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing experiment results were overwritten.")
    write_experiment_results(experiment_rows, output_path, overwrite=True)

    cbf_row = make_experiment_result_row(seed=14)
    cbf_row["algorithm"] = "CBF"
    cbf_row["final_population_best_fitness"] = 13.0
    cbf_output_path = temporary_root / "cbf-results.csv"
    write_experiment_results([cbf_row], cbf_output_path)
    loaded_cbf_row = read_experiment_results(cbf_output_path)[0]
    assert loaded_cbf_row["algorithm"] == "CBF"
    assert loaded_cbf_row["final_population_best_fitness"] == "13.0"
    assert float(loaded_cbf_row["best_fitness"]) <= float(
        loaded_cbf_row["final_population_best_fitness"]
    )

    bf_with_final_population_fitness = make_experiment_result_row(seed=15)
    bf_with_final_population_fitness["final_population_best_fitness"] = 13.0
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [bf_with_final_population_fitness],
            temporary_root / "bf-with-final-fitness.csv",
        )
    )

    cbf_without_final_population_fitness = make_experiment_result_row(seed=16)
    cbf_without_final_population_fitness["algorithm"] = "CBF"
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [cbf_without_final_population_fitness],
            temporary_root / "cbf-without-final-fitness.csv",
        )
    )

    cbf_final_fitness_too_low = dict(cbf_row)
    cbf_final_fitness_too_low["seed"] = 17
    cbf_final_fitness_too_low["final_population_best_fitness"] = 12.0
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [cbf_final_fitness_too_low],
            temporary_root / "cbf-inconsistent-fitness.csv",
        )
    )

    unknown_algorithm_row = make_experiment_result_row(seed=18)
    unknown_algorithm_row["algorithm"] = "KMeans"
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [unknown_algorithm_row],
            temporary_root / "unknown-algorithm.csv",
        )
    )

    failed_row = make_experiment_result_row(seed=9)
    failed_row["run_status"] = "failed"
    failed_row["failure_description"] = "Optimizer did not converge."
    for metric in (
        "best_fitness",
        "final_population_best_fitness",
        "silhouette",
        "error_rate_percent",
        "accuracy_percent",
        "correct_count",
        "incorrect_count",
        "nonempty_cluster_count",
        "empty_cluster_rejections",
        "runtime_seconds",
        "fitness_evaluation_count",
        "accepted_movements",
        "reproduction_events",
        "elimination_events",
        "dispersed_bacteria_count",
    ):
        failed_row[metric] = None

    failed_output_path = temporary_root / "failed-results.csv"
    write_experiment_results([failed_row], failed_output_path)
    loaded_failed_row = read_experiment_results(failed_output_path)[0]
    assert loaded_failed_row["run_status"] == "failed"
    assert loaded_failed_row["failure_description"] == (
        "Optimizer did not converge."
    )
    assert loaded_failed_row["best_fitness"] is None
    assert loaded_failed_row["matplotlib_version"] == "3.9.0"

    duplicate_rows = [
        make_experiment_result_row(),
        make_experiment_result_row(),
    ]
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            duplicate_rows,
            temporary_root / "duplicates.csv",
        )
    )
    assert_experiment_results_value_error(
        lambda: write_experiment_results([], temporary_root / "empty.csv")
    )

    missing_field_row = make_experiment_result_row()
    del missing_field_row["silhouette"]
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [missing_field_row],
            temporary_root / "missing.csv",
        )
    )

    unexpected_field_row = make_experiment_result_row()
    unexpected_field_row["extra"] = "not allowed"
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [unexpected_field_row],
            temporary_root / "unexpected.csv",
        )
    )

    success_with_failure_text = make_experiment_result_row(seed=10)
    success_with_failure_text["failure_description"] = "Unexpected failure."
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [success_with_failure_text],
            temporary_root / "success-with-failure-text.csv",
        )
    )

    failed_without_failure_text = make_experiment_result_row(seed=11)
    failed_without_failure_text["run_status"] = "failed"
    failed_without_failure_text["failure_description"] = ""
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [failed_without_failure_text],
            temporary_root / "failed-without-description.csv",
        )
    )

    invalid_status_row = make_experiment_result_row(seed=12)
    invalid_status_row["run_status"] = "skipped"
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [invalid_status_row],
            temporary_root / "invalid-status.csv",
        )
    )

    success_without_metric = make_experiment_result_row(seed=13)
    success_without_metric["best_fitness"] = None
    assert_experiment_results_value_error(
        lambda: write_experiment_results(
            [success_without_metric],
            temporary_root / "success-without-metric.csv",
        )
    )

    duplicate_input = temporary_root / "duplicate-input.csv"
    with duplicate_input.open("w", encoding="utf-8", newline="") as result_file:
        csv_writer = csv.writer(result_file)
        csv_writer.writerow(EXPERIMENT_RESULT_COLUMNS)
        for row in duplicate_rows:
            csv_writer.writerow(
                [
                    row[column]
                    if column != "parameter_configuration"
                    else '{"num_clusters":3,"step_size":0.1}'
                    for column in EXPERIMENT_RESULT_COLUMNS
                ]
            )
    assert_experiment_results_value_error(
        lambda: read_experiment_results(duplicate_input)
    )

    malformed_input = temporary_root / "malformed.csv"
    malformed_input.write_text("wrong,header\n", encoding="utf-8")
    assert_experiment_results_value_error(
        lambda: read_experiment_results(malformed_input)
    )

print("Experiment CSV tests passed.")


import subprocess
from contextlib import redirect_stderr

import matplotlib
import scipy
import sklearn

import bf_reusable
import data_loader
from run_prototype_experiments import (
    DATASET_CLUSTER_COUNTS,
    MissingRequiredCounterError,
    describe_failure,
    execute_one_run,
    execute_runs,
    get_reproducibility_metadata,
    main as run_prototype_main,
    parameter_configuration,
    parse_arguments,
    planned_runs,
)


def repository_status():
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def run_prototype_quietly(argv):
    with redirect_stdout(io.StringIO()) as captured_output:
        with redirect_stderr(io.StringIO()):
            exit_code = run_prototype_main(argv)
    return exit_code, captured_output.getvalue()


def assert_cli_rejected(argv):
    try:
        with redirect_stderr(io.StringIO()):
            parse_arguments(argv)
    except SystemExit as error:
        assert error.code != 0
        return
    raise AssertionError(f"Expected command-line rejection for {argv}.")


def make_fake_runner(calls, algorithm, **overrides):
    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))
        labels = np.array([1, 1, 1, 0, 0, 0])
        centers = np.array([[1.0, 1.0], [-1.0, -1.0]])
        result = {
            "best_fitness": 2.0,
            "silhouette": 0.5,
            "runtime_seconds": 0.01,
            "fitness_evaluation_count": 10,
            "labels": labels,
            "centers": centers,
            "accepted_movements": 4,
            "reproduction_events": 1,
            "elimination_events": 1,
            "dispersed_bacteria_count": 2,
            "empty_cluster_rejections": 0,
        }
        if algorithm == "CBF":
            result["final_population_best_fitness"] = 3.0
            result["belief_best_bacterium"] = centers.copy()
        result.update(overrides)
        return result

    return fake_runner


status_before_runner_tests = repository_status()

default_runs = planned_runs(parse_arguments(["--dry-run"]))
assert len(default_runs) == 240
assert len(set(default_runs)) == 240
assert {dataset for dataset, _, _ in default_runs} == {
    "iris",
    "wine",
    "glass",
    "cancer",
}
assert {algorithm for _, algorithm, _ in default_runs} == {"BF", "CBF"}
assert {seed for _, _, seed in default_runs} == set(range(30))

smoke_runs = planned_runs(
    parse_arguments(["--dry-run", "--datasets", "iris", "--seeds", "0", "1"])
)
assert smoke_runs == (
    ("iris", "BF", 0),
    ("iris", "BF", 1),
    ("iris", "CBF", 0),
    ("iris", "CBF", 1),
)

with tempfile.TemporaryDirectory() as temporary_directory:
    temporary_root = Path(temporary_directory)
    existing_output = temporary_root / "existing.csv"
    existing_output.write_text("keep me\n", encoding="utf-8")

    for rejected_arguments in (
        [],
        ["--output", str(existing_output)],
        ["--dry-run", "--datasets"],
        ["--dry-run", "--datasets", "breast_cancer"],
        ["--dry-run", "--datasets", "iris", "iris"],
        ["--dry-run", "--algorithms", "KMeans"],
        ["--dry-run", "--algorithms", "BF", "BF"],
        ["--dry-run", "--seeds", "1", "1"],
        ["--dry-run", "--seeds", "-1"],
        ["--dry-run", "--seeds", "1.5"],
    ):
        assert_cli_rejected(rejected_arguments)

    # Dry-run must not load datasets, execute algorithms, or create files.
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Dry-run must not load data or run algorithms.")

    original_loader = data_loader.load_standardized_dataset
    original_run_bf = bf_reusable.run_bf
    data_loader.load_standardized_dataset = forbidden_call
    bf_reusable.run_bf = forbidden_call
    try:
        dry_run_output = temporary_root / "dry-run.csv"
        exit_code, printed_plan = run_prototype_quietly(
            ["--dry-run", "--output", str(dry_run_output)]
        )
        assert exit_code == 0
        assert printed_plan.strip().endswith("Total planned runs: 240")
        assert not dry_run_output.exists()

        exit_code, _ = run_prototype_quietly(
            ["--dry-run", "--output", str(existing_output)]
        )
        assert exit_code == 0
        assert existing_output.read_text(encoding="utf-8") == "keep me\n"
    finally:
        data_loader.load_standardized_dataset = original_loader
        bf_reusable.run_bf = original_run_bf
    assert [path.name for path in temporary_root.iterdir()] == ["existing.csv"]

for dataset in DATASET_CLUSTER_COUNTS:
    bf_configuration = parameter_configuration("BF", dataset, 5)
    cbf_configuration = parameter_configuration("CBF", dataset, 5)
    assert bf_configuration["num_clusters"] == DATASET_CLUSTER_COUNTS[dataset]
    assert bf_configuration["random_seed"] == 5
    assert set(cbf_configuration) - set(bf_configuration) == {
        "cultural_influence",
        "acceptance_ratio",
    }

runner_metadata = get_reproducibility_metadata()
assert set(runner_metadata) == {
    "git_commit",
    "python_version",
    "numpy_version",
    "sklearn_version",
    "scipy_version",
    "matplotlib_version",
}
assert len(runner_metadata["git_commit"]) == 40
assert set(runner_metadata["git_commit"]) <= set("0123456789abcdef")
assert runner_metadata["numpy_version"] == np.__version__
assert runner_metadata["sklearn_version"] == sklearn.__version__
assert runner_metadata["scipy_version"] == scipy.__version__
assert runner_metadata["matplotlib_version"] == matplotlib.__version__

fake_X = np.array(
    [
        [1.0, 1.1],
        [0.9, 1.0],
        [1.1, 0.9],
        [-1.0, -1.1],
        [-0.9, -1.0],
        [-1.1, -0.9],
    ]
)
fake_y = np.array([0, 0, 0, 1, 1, 1])
fake_X_scaled = fake_X.copy()
fake_dataset = (fake_X, fake_y, fake_X_scaled, ["a", "b"], ["first", "second"])


def run_fake(algorithm, **overrides):
    calls = []
    configuration = parameter_configuration(algorithm, "cancer", 3)
    row = execute_one_run(
        "cancer",
        algorithm,
        3,
        fake_dataset,
        configuration,
        runner_metadata,
        {algorithm: make_fake_runner(calls, algorithm, **overrides)},
        np,
        clustering_error_rate,
    )
    return row, calls, configuration


fake_rows = []
for algorithm in ("BF", "CBF"):
    fake_row, fake_calls, fake_configuration = run_fake(algorithm)
    fake_rows.append(fake_row)
    assert len(fake_calls) == 1
    call_args, call_kwargs = fake_calls[0]

    # Only the standardized features reach the optimizer; labels do not.
    assert len(call_args) == 1
    assert call_args[0] is fake_X_scaled
    assert all(value is not fake_y for value in call_kwargs.values())
    # The recorded configuration must equal the arguments actually used.
    assert call_kwargs == fake_configuration

    assert set(fake_row) == set(EXPERIMENT_RESULT_COLUMNS)
    assert fake_row["run_status"] == "success"
    assert fake_row["failure_description"] == ""
    assert fake_row["parameter_configuration"] == fake_configuration
    assert fake_row["nonempty_cluster_count"] == 2
    # Cluster identifiers are swapped relative to classes; mapping fixes this.
    assert fake_row["accuracy_percent"] == 100.0
    assert fake_row["error_rate_percent"] == 0.0
    assert fake_row["correct_count"] == 6
    assert fake_row["incorrect_count"] == 0
    for field, value in runner_metadata.items():
        assert fake_row[field] == value

assert fake_rows[0]["final_population_best_fitness"] is None
assert fake_rows[1]["final_population_best_fitness"] == 3.0
assert fake_rows[1]["best_fitness"] <= fake_rows[1][
    "final_population_best_fitness"
]

for algorithm, overrides in (
    ("CBF", {"final_population_best_fitness": 1.0}),
    ("CBF", {"belief_best_bacterium": np.array([[0.0, 0.0], [1.0, 1.0]])}),
    ("BF", {"labels": np.zeros(6, dtype=int)}),
    ("BF", {"labels": np.array([0, 1, 0])}),
    ("BF", {"centers": np.zeros((3, 2))}),
    ("BF", {"best_fitness": float("inf")}),
):
    try:
        run_fake(algorithm, **overrides)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid {algorithm} result was accepted.")

for algorithm, overrides in (
    ("BF", {"fitness_evaluation_count": None}),
    ("CBF", {"accepted_movements": None}),
):
    try:
        run_fake(algorithm, **overrides)
    except MissingRequiredCounterError:
        pass
    else:
        raise AssertionError("Unavailable required counter was accepted.")


def run_fake_without_counter(*args, **kwargs):
    result = make_fake_runner([], "BF")(*args, **kwargs)
    del result["dispersed_bacteria_count"]
    return result


try:
    execute_one_run(
        "cancer",
        "BF",
        3,
        fake_dataset,
        parameter_configuration("BF", "cancer", 3),
        runner_metadata,
        {"BF": run_fake_without_counter},
        np,
        clustering_error_rate,
    )
except MissingRequiredCounterError:
    pass
else:
    raise AssertionError("Missing required counter was accepted.")

assert describe_failure(ValueError("first\n   second")) == (
    "ValueError: first second"
)
assert describe_failure(RuntimeError()) == (
    "RuntimeError: No exception message was provided."
)
long_description = describe_failure(ValueError("x" * 1000))
assert len(long_description) == 500
assert long_description.endswith("...")

loader_calls = []


def failing_loader(dataset_name):
    loader_calls.append(dataset_name)
    raise OSError("dataset source unavailable")


def failing_run_bf(*args, **kwargs):
    raise RuntimeError("optimizer crashed")


original_loader = data_loader.load_standardized_dataset
original_run_bf = bf_reusable.run_bf
try:
    data_loader.load_standardized_dataset = failing_loader
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        load_failure_rows = execute_runs(
            (("glass", "BF", 0), ("glass", "CBF", 0)),
            runner_metadata,
        )
    data_loader.load_standardized_dataset = original_loader

    bf_reusable.run_bf = failing_run_bf
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        runner_failure_rows = execute_runs(
            (("iris", "BF", 4),),
            runner_metadata,
        )
finally:
    data_loader.load_standardized_dataset = original_loader
    bf_reusable.run_bf = original_run_bf

# A dataset that fails to load is attempted once; every planned run is kept.
assert loader_calls == ["glass"]
assert [row["failure_description"] for row in load_failure_rows] == [
    "OSError: dataset source unavailable",
    "OSError: dataset source unavailable",
]
assert runner_failure_rows[0]["failure_description"] == (
    "RuntimeError: optimizer crashed"
)
failure_rows = load_failure_rows + runner_failure_rows
assert [
    (row["dataset"], row["algorithm"], row["seed"]) for row in failure_rows
] == [("glass", "BF", 0), ("glass", "CBF", 0), ("iris", "BF", 4)]
for failure_row in failure_rows:
    assert set(failure_row) == set(EXPERIMENT_RESULT_COLUMNS)
    assert failure_row["run_status"] == "failed"
    assert failure_row["best_fitness"] is None
    assert failure_row["runtime_seconds"] is None
    assert failure_row["fitness_evaluation_count"] is None
    assert failure_row["git_commit"] == runner_metadata["git_commit"]

with tempfile.TemporaryDirectory() as temporary_directory:
    temporary_root = Path(temporary_directory)
    write_experiment_results(failure_rows, temporary_root / "failures.csv")
    assert len(read_experiment_results(temporary_root / "failures.csv")) == 3

    smoke_arguments = ["--datasets", "iris", "--seeds", "0", "1"]
    smoke_outputs = (
        temporary_root / "smoke-1.csv",
        temporary_root / "smoke-2.csv",
    )
    for smoke_output in smoke_outputs:
        exit_code, _ = run_prototype_quietly(
            smoke_arguments + ["--output", str(smoke_output)]
        )
        assert exit_code == 0

    first_smoke_rows = read_experiment_results(smoke_outputs[0])
    second_smoke_rows = read_experiment_results(smoke_outputs[1])
    assert len(first_smoke_rows) == 4
    assert len(second_smoke_rows) == 4
    assert [
        (row["dataset"], row["algorithm"], row["seed"])
        for row in first_smoke_rows
    ] == [
        ("iris", "BF", "0"),
        ("iris", "BF", "1"),
        ("iris", "CBF", "0"),
        ("iris", "CBF", "1"),
    ]
    for first_row, second_row in zip(first_smoke_rows, second_smoke_rows):
        assert first_row["run_status"] == "success"
        assert first_row["nonempty_cluster_count"] == "3"
        assert first_row["git_commit"] == runner_metadata["git_commit"]
        assert first_row["parameter_configuration"] == parameter_configuration(
            first_row["algorithm"],
            first_row["dataset"],
            int(first_row["seed"]),
        )
        if first_row["algorithm"] == "BF":
            assert first_row["final_population_best_fitness"] is None
        else:
            assert float(first_row["best_fitness"]) <= float(
                first_row["final_population_best_fitness"]
            )
        # Runtime is environment-dependent; every other field must repeat.
        for column in EXPERIMENT_RESULT_COLUMNS:
            if column != "runtime_seconds":
                assert first_row[column] == second_row[column], column

    protected_contents = smoke_outputs[0].read_bytes()
    assert_cli_rejected(smoke_arguments + ["--output", str(smoke_outputs[0])])
    assert smoke_outputs[0].read_bytes() == protected_contents
    exit_code, _ = run_prototype_quietly(
        smoke_arguments + ["--output", str(smoke_outputs[0]), "--overwrite"]
    )
    assert exit_code == 0
    assert len(read_experiment_results(smoke_outputs[0])) == 4

assert repository_status() == status_before_runner_tests

print("Prototype runner tests passed.")
