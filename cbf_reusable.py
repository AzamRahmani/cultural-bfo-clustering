import numpy as np

from data_loader import load_standardized_dataset
from sklearn.metrics import silhouette_score


def calculate_fitness(
    X_scaled,
    centers,
    require_all_clusters=True,
):
    distances = np.linalg.norm(
        X_scaled[:, np.newaxis, :]
        - centers[np.newaxis, :, :],
        axis=2,
    )

    labels = np.argmin(
        distances,
        axis=1,
    )

    # Temporary prototype policy: reject solutions containing
    # empty clusters. The exact thesis policy remains unverified.
    if require_all_clusters:
        nonempty_cluster_count = len(
            np.unique(labels)
        )

        if nonempty_cluster_count < len(centers):
            return np.inf, labels

    fitness = np.sum(
        (
            X_scaled
            - centers[labels]
        ) ** 2
    )

    return fitness, labels


def run_cbf(
    X_scaled,
    num_clusters,
    random_seed=42,
    population_size=20,
    num_iterations=50,
    step_size=0.1,
    max_swim_steps=3,
    reproduction_interval=10,
    elimination_interval=10,
    elimination_probability=0.10,
    cultural_influence=0.05,
    acceptance_ratio=0.35,
):
    """
    Run a reusable exploratory CBF prototype on standardized data.

    X_scaled is a standardized two-dimensional feature matrix, and
    num_clusters is the requested number of clusters. random_seed controls
    reproducibility. The function combines simplified BF with situational
    and normative knowledge and returns a dictionary of results. This is
    not the complete thesis CBF implementation.
    """
    if X_scaled.ndim != 2:
        raise ValueError(
            "X_scaled must be a two-dimensional array."
        )

    if num_clusters < 2:
        raise ValueError(
            "num_clusters must be at least 2."
        )

    if num_clusters > X_scaled.shape[0]:
        raise ValueError(
            "num_clusters cannot exceed the number of samples."
        )

    if population_size < 2:
        raise ValueError(
            "population_size must be at least 2."
        )

    if not 0 < acceptance_ratio <= 1:
        raise ValueError(
            "acceptance_ratio must be greater than 0 and at most 1."
        )

    if cultural_influence < 0:
        raise ValueError(
            "cultural_influence cannot be negative."
        )

    rng = np.random.default_rng(random_seed)

    feature_min = X_scaled.min(axis=0)
    feature_max = X_scaled.max(axis=0)
    number_of_features = X_scaled.shape[1]

    bacteria = rng.uniform(
        low=feature_min,
        high=feature_max,
        size=(
            population_size,
            num_clusters,
            number_of_features,
        ),
    )

    fitness_values = np.zeros(population_size)
    for index in range(population_size):
        fitness_values[index], _ = calculate_fitness(
            X_scaled,
            bacteria[index],
        )

    if not np.any(np.isfinite(fitness_values)):
        raise RuntimeError(
            "The initial population contains no valid "
            "clustering solution."
        )

    belief_best_index = np.argmin(fitness_values)
    belief_best_bacterium = bacteria[belief_best_index].copy()
    belief_best_fitness = float(
        fitness_values[belief_best_index]
    )
    belief_updates = 0

    accepted_count = max(
        1,
        int(
            np.ceil(
                population_size
                * acceptance_ratio
            )
        ),
    )

    initial_accepted_indices = np.argsort(
        fitness_values
    )[:accepted_count]
    initial_accepted_bacteria = bacteria[
        initial_accepted_indices
    ]

    normative_lower = initial_accepted_bacteria.min(
        axis=0
    )
    normative_upper = initial_accepted_bacteria.max(
        axis=0
    )

    normative_updates = 0
    normative_bounds_changes = 0
    normative_clipped_candidates = 0
    cultural_movements = 0

    health_values = np.zeros(population_size)
    accepted_movements = 0
    health_accumulation_steps = 0
    reproduction_events = 0
    elimination_events = 0
    dispersed_bacteria_count = 0
    empty_cluster_rejections = 0

    for iteration in range(1, num_iterations + 1):
        for index in range(population_size):
            direction = rng.normal(
                size=bacteria[index].shape
            )
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                continue

            direction = direction / direction_norm

            cultural_direction = (
                belief_best_bacterium
                - bacteria[index]
            )
            cultural_direction_norm = np.linalg.norm(
                cultural_direction
            )
            if cultural_direction_norm > 0:
                cultural_direction = (
                    cultural_direction
                    / cultural_direction_norm
                )

            combined_direction = (
                direction
                + cultural_influence
                * cultural_direction
            )
            combined_direction_norm = np.linalg.norm(
                combined_direction
            )
            if combined_direction_norm == 0:
                continue

            combined_direction = (
                combined_direction
                / combined_direction_norm
            )

            for _ in range(max_swim_steps):
                candidate = (
                    bacteria[index]
                    + step_size
                    * combined_direction
                )
                clipped_candidate = np.clip(
                    candidate,
                    normative_lower,
                    normative_upper,
                )
                if not np.array_equal(
                    candidate,
                    clipped_candidate,
                ):
                    normative_clipped_candidates += 1

                candidate = clipped_candidate
                candidate_fitness, _ = calculate_fitness(
                    X_scaled,
                    candidate,
                )

                if not np.isfinite(candidate_fitness):
                    empty_cluster_rejections += 1
                    break

                if candidate_fitness < fitness_values[index]:
                    bacteria[index] = candidate
                    fitness_values[index] = candidate_fitness
                    accepted_movements += 1
                    cultural_movements += 1
                    continue

                break

        current_best_index = np.argmin(fitness_values)
        current_best_fitness = fitness_values[
            current_best_index
        ]
        if current_best_fitness < belief_best_fitness:
            belief_best_bacterium = bacteria[
                current_best_index
            ].copy()
            belief_best_fitness = float(
                current_best_fitness
            )
            belief_updates += 1

        accepted_indices = np.argsort(
            fitness_values
        )[:accepted_count]
        accepted_bacteria = bacteria[
            accepted_indices
        ]
        new_normative_lower = accepted_bacteria.min(
            axis=0
        )
        new_normative_upper = accepted_bacteria.max(
            axis=0
        )
        bounds_changed = (
            not np.array_equal(
                normative_lower,
                new_normative_lower,
            )
            or not np.array_equal(
                normative_upper,
                new_normative_upper,
            )
        )
        if bounds_changed:
            normative_bounds_changes += 1

        normative_lower = new_normative_lower.copy()
        normative_upper = new_normative_upper.copy()
        normative_updates += 1

        health_values += fitness_values
        health_accumulation_steps += 1

        if iteration % reproduction_interval == 0:
            sorted_indices = np.argsort(
                health_values
            )
            half_population = population_size // 2
            best_indices = sorted_indices[
                :half_population
            ]
            best_bacteria = bacteria[
                best_indices
            ].copy()
            best_fitness_values = fitness_values[
                best_indices
            ].copy()

            bacteria = np.concatenate(
                [
                    best_bacteria,
                    best_bacteria.copy(),
                ],
                axis=0,
            )

            fitness_values = np.concatenate(
                [
                    best_fitness_values,
                    best_fitness_values.copy(),
                ],
                axis=0,
            )

            health_values = np.zeros(
                population_size
            )
            reproduction_events += 1

        if iteration % elimination_interval == 0:
            protected_best_index = np.argmin(
                fitness_values
            )
            for index in range(population_size):
                if index == protected_best_index:
                    continue

                if rng.random() < elimination_probability:
                    replacement_attempts = 0
                    while replacement_attempts < 100:
                        replacement_attempts += 1
                        replacement = rng.uniform(
                            low=feature_min,
                            high=feature_max,
                            size=(
                                num_clusters,
                                number_of_features,
                            ),
                        )
                        replacement_fitness, _ = (
                            calculate_fitness(
                                X_scaled,
                                replacement,
                            )
                        )
                        if np.isfinite(replacement_fitness):
                            bacteria[index] = replacement
                            fitness_values[index] = (
                                replacement_fitness
                            )
                            break

                        empty_cluster_rejections += 1
                    else:
                        raise RuntimeError(
                            "Could not generate a valid dispersed "
                            "bacterium."
                        )
                    dispersed_bacteria_count += 1

            elimination_events += 1

    final_population_best_index = np.argmin(
        fitness_values
    )

    final_population_best_fitness = float(
        fitness_values[final_population_best_index]
    )

    best_bacterium = belief_best_bacterium.copy()
    best_fitness, best_labels = calculate_fitness(
        X_scaled,
        best_bacterium,
    )
    silhouette = silhouette_score(
        X_scaled,
        best_labels,
    )
    belief_difference = abs(
        best_fitness
        - belief_best_fitness
    )

    return {
        "best_fitness": float(best_fitness),
        "final_population_best_fitness": (
            final_population_best_fitness
        ),
        "silhouette": float(silhouette),
        "labels": best_labels,
        "centers": best_bacterium,
        "population": bacteria,
        "fitness_values": fitness_values,
        "health_values": health_values,
        "belief_best_bacterium": belief_best_bacterium,
        "belief_best_fitness": float(
            belief_best_fitness
        ),
        "belief_difference": float(
            belief_difference
        ),
        "normative_lower": normative_lower,
        "normative_upper": normative_upper,
        "accepted_count": accepted_count,
        "accepted_movements": accepted_movements,
        "cultural_movements": cultural_movements,
        "belief_updates": belief_updates,
        "normative_updates": normative_updates,
        "normative_bounds_changes": (
            normative_bounds_changes
        ),
        "normative_clipped_candidates": (
            normative_clipped_candidates
        ),
        "health_accumulation_steps": (
            health_accumulation_steps
        ),
        "reproduction_events": reproduction_events,
        "elimination_events": elimination_events,
        "dispersed_bacteria_count": (
            dispersed_bacteria_count
        ),
        "empty_cluster_rejections": (
            empty_cluster_rejections
        ),
        "random_seed": random_seed,
    }


def print_result(dataset_name, result):
    print(f"Reusable CBF: {dataset_name}")
    print(f"Fitness: {result['best_fitness']:.4f}")
    print(f"Silhouette: {result['silhouette']:.4f}")
    print(f"Labels shape: {result['labels'].shape}")
    print(f"Centers shape: {result['centers'].shape}")
    print(
        f"Population shape: "
        f"{result['population'].shape}"
    )
    print(
        f"Belief fitness: "
        f"{result['belief_best_fitness']:.4f}"
    )
    print(
        "Final population best fitness: "
        f"{result['final_population_best_fitness']:.4f}"
    )
    print(
        f"Belief bacterium shape: "
        f"{result['belief_best_bacterium'].shape}"
    )
    print(
        f"Normative lower shape: "
        f"{result['normative_lower'].shape}"
    )
    print(
        f"Normative upper shape: "
        f"{result['normative_upper'].shape}"
    )
    print(
        f"Accepted bacteria: "
        f"{result['accepted_count']}"
    )
    print(
        f"Resulting clusters: "
        f"{len(np.unique(result['labels']))}"
    )
    print(
        f"Accepted movements: "
        f"{result['accepted_movements']}"
    )
    print(
        f"Cultural movements: "
        f"{result['cultural_movements']}"
    )
    print(
        f"Belief updates: "
        f"{result['belief_updates']}"
    )
    print(
        f"Normative updates: "
        f"{result['normative_updates']}"
    )
    print(
        f"Normative bounds changes: "
        f"{result['normative_bounds_changes']}"
    )
    print(
        f"Normative clipped candidates: "
        f"{result['normative_clipped_candidates']}"
    )
    print(
        f"Health accumulation steps: "
        f"{result['health_accumulation_steps']}"
    )
    print(
        f"Reproduction events: "
        f"{result['reproduction_events']}"
    )
    print(
        f"Elimination events: "
        f"{result['elimination_events']}"
    )
    print(
        f"Dispersed bacteria count: "
        f"{result['dispersed_bacteria_count']}"
    )
    print(
        f"Empty-cluster rejections: "
        f"{result['empty_cluster_rejections']}"
    )
    print(f"Random seed: {result['random_seed']}")
    print()


def assert_common_result(result, labels_shape, center_shape):
    assert result["labels"].shape == labels_shape
    assert result["centers"].shape == center_shape
    assert result["population"].shape == (
        20,
        *center_shape,
    )
    assert result["belief_best_bacterium"].shape == center_shape
    assert result["normative_lower"].shape == center_shape
    assert result["normative_upper"].shape == center_shape
    assert result["fitness_values"].shape == (20,)
    assert np.all(
        np.isfinite(result["fitness_values"])
    )
    assert result["health_values"].shape == (20,)
    assert result["accepted_count"] == 7
    assert np.isfinite(result["best_fitness"])
    assert result["best_fitness"] > 0
    assert np.isfinite(
        result["final_population_best_fitness"]
    )
    assert (
        result["best_fitness"]
        <= result["final_population_best_fitness"] + 1e-12
    )
    assert np.isfinite(result["belief_best_fitness"])
    assert -1 <= result["silhouette"] <= 1
    assert len(np.unique(result["labels"])) == 3
    assert result["empty_cluster_rejections"] >= 0
    assert result["health_accumulation_steps"] == 50
    assert result["reproduction_events"] == 5
    assert result["elimination_events"] == 5
    assert result["normative_updates"] == 50
    assert len(result["population"]) == 20
    assert np.all(
        result["normative_lower"]
        <= result["normative_upper"]
    )
    assert (
        result["belief_best_fitness"]
        <= result["best_fitness"] + 1e-12
    )
    assert result["belief_difference"] < 1e-10


if __name__ == "__main__":
    _, _, iris_scaled, _, _ = (
        load_standardized_dataset("iris")
    )
    iris_result = run_cbf(
        iris_scaled,
        num_clusters=3,
        random_seed=42,
    )
    print_result("Iris", iris_result)

    assert_common_result(
        iris_result,
        (150,),
        (3, 4),
    )

    _, _, wine_scaled, _, _ = (
        load_standardized_dataset("wine")
    )
    wine_result = run_cbf(
        wine_scaled,
        num_clusters=3,
        random_seed=42,
    )
    print_result("Wine", wine_result)

    assert_common_result(
        wine_result,
        (178,),
        (3, 13),
    )

    print("Reusable CBF checks passed.")

    try:
        run_cbf(
            iris_scaled,
            num_clusters=1,
        )
    except ValueError as error:
        print(f"Expected error: {error}")
    else:
        raise AssertionError(
            "Invalid cluster count did not raise ValueError."
        )

    try:
        run_cbf(
            iris_scaled,
            num_clusters=3,
            acceptance_ratio=0,
        )
    except ValueError as error:
        print(f"Expected error: {error}")
    else:
        raise AssertionError(
            "Invalid acceptance ratio did not raise ValueError."
        )

    try:
        run_cbf(
            iris_scaled,
            num_clusters=3,
            cultural_influence=-0.1,
        )
    except ValueError as error:
        print(f"Expected error: {error}")
    else:
        raise AssertionError(
            "Invalid cultural influence did not raise ValueError."
        )
