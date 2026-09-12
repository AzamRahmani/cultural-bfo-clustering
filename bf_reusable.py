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


def run_bf(
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
):
    """
    Run a simplified Bacterial Foraging prototype on standardized data.

    X_scaled is the standardized feature matrix. num_clusters is the
    requested number of clusters. random_seed controls reproducibility.
    The function returns the best result and run counters. This remains
    a simplified BF prototype, not a complete thesis implementation.
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

    rng = np.random.default_rng(random_seed)

    feature_min = X_scaled.min(axis=0)
    feature_max = X_scaled.max(axis=0)
    number_of_features = X_scaled.shape[1]

    bacteria = rng.uniform(
        low=feature_min,
        high=feature_max,
        size=(population_size, num_clusters, number_of_features),
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

    health_values = np.zeros(population_size)
    accepted_movements = 0
    health_accumulation_steps = 0
    reproduction_events = 0
    elimination_events = 0
    dispersed_bacteria_count = 0
    empty_cluster_rejections = 0

    for iteration in range(1, num_iterations + 1):
        for index in range(population_size):
            direction = rng.normal(size=bacteria[index].shape)
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                continue

            direction = direction / direction_norm

            for _ in range(max_swim_steps):
                candidate = bacteria[index] + step_size * direction
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
                    continue

                break

        health_values += fitness_values
        health_accumulation_steps += 1

        if iteration % reproduction_interval == 0:
            sorted_indices = np.argsort(health_values)
            half_population = population_size // 2
            best_indices = sorted_indices[:half_population]
            best_bacteria = bacteria[best_indices].copy()
            best_fitness_values = fitness_values[best_indices].copy()

            bacteria = np.concatenate(
                [best_bacteria, best_bacteria.copy()],
                axis=0,
            )

            fitness_values = np.concatenate(
                [best_fitness_values, best_fitness_values.copy()],
                axis=0,
            )
            health_values = np.zeros(population_size)
            reproduction_events += 1

        if iteration % elimination_interval == 0:
            protected_best_index = np.argmin(fitness_values)
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
                            size=(num_clusters, number_of_features),
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

    best_index = np.argmin(fitness_values)
    best_bacterium = bacteria[best_index]
    best_fitness, best_labels = calculate_fitness(
        X_scaled,
        best_bacterium,
    )
    silhouette = silhouette_score(X_scaled, best_labels)

    return {
        "best_fitness": float(best_fitness),
        "silhouette": float(silhouette),
        "labels": best_labels,
        "centers": best_bacterium,
        "population": bacteria,
        "fitness_values": fitness_values,
        "health_values": health_values,
        "accepted_movements": accepted_movements,
        "health_accumulation_steps": health_accumulation_steps,
        "reproduction_events": reproduction_events,
        "elimination_events": elimination_events,
        "dispersed_bacteria_count": dispersed_bacteria_count,
        "empty_cluster_rejections": (
            empty_cluster_rejections
        ),
        "random_seed": random_seed,
    }


def print_result(dataset_name, result):
    print(f"Reusable BF: {dataset_name}")
    print(f"Fitness: {result['best_fitness']:.4f}")
    print(f"Silhouette: {result['silhouette']:.4f}")
    print(f"Labels shape: {result['labels'].shape}")
    print(f"Centers shape: {result['centers'].shape}")
    print(f"Population shape: {result['population'].shape}")
    print(f"Resulting clusters: {len(np.unique(result['labels']))}")
    print(f"Accepted movements: {result['accepted_movements']}")
    print(
        "Health accumulation steps: "
        f"{result['health_accumulation_steps']}"
    )
    print(f"Reproduction events: {result['reproduction_events']}")
    print(f"Elimination events: {result['elimination_events']}")
    print(
        "Dispersed bacteria count: "
        f"{result['dispersed_bacteria_count']}"
    )
    print(
        "Empty-cluster rejections: "
        f"{result['empty_cluster_rejections']}"
    )
    print()


if __name__ == "__main__":
    _, _, iris_scaled, _, _ = (
        load_standardized_dataset("iris")
    )

    iris_result = run_bf(
        iris_scaled,
        num_clusters=3,
        random_seed=42,
    )

    print_result("Iris", iris_result)

    assert iris_result["labels"].shape == (150,)
    assert iris_result["centers"].shape == (3, 4)
    assert iris_result["population"].shape == (20, 3, 4)
    assert len(np.unique(iris_result["labels"])) == 3
    assert np.isfinite(iris_result["best_fitness"])
    assert np.all(
        np.isfinite(iris_result["fitness_values"])
    )
    assert iris_result["empty_cluster_rejections"] >= 0
    assert -1 <= iris_result["silhouette"] <= 1
    assert iris_result["health_accumulation_steps"] == 50
    assert iris_result["reproduction_events"] == 5
    assert iris_result["elimination_events"] == 5

    _, _, wine_scaled, _, _ = (
        load_standardized_dataset("wine")
    )

    wine_result = run_bf(
        wine_scaled,
        num_clusters=3,
        random_seed=42,
    )

    print_result("Wine", wine_result)

    assert wine_result["labels"].shape == (178,)
    assert wine_result["centers"].shape == (3, 13)
    assert wine_result["population"].shape == (20, 3, 13)
    assert len(np.unique(wine_result["labels"])) == 3
    assert np.isfinite(wine_result["best_fitness"])
    assert np.all(
        np.isfinite(wine_result["fitness_values"])
    )
    assert wine_result["empty_cluster_rejections"] >= 0
    assert -1 <= wine_result["silhouette"] <= 1
    assert wine_result["health_accumulation_steps"] == 50
    assert wine_result["reproduction_events"] == 5
    assert wine_result["elimination_events"] == 5

    print("Reusable BF checks passed.")

    try:
        run_bf(
            iris_scaled,
            num_clusters=1,
        )
    except ValueError as error:
        print(f"Expected error: {error}")
    else:
        raise AssertionError(
            "Invalid cluster count did not raise ValueError."
        )
