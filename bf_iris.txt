import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# This is a minimal exploratory prototype, not the complete thesis BF or CBF algorithm.
# This version implements simplified tumble-and-swim,
# bacterial health accumulation, reproduction, and elimination-dispersal.
# It does not yet include cultural knowledge or the full thesis CBF method.

# Set the random generator with fixed seed for reproducibility
rng = np.random.default_rng(42)

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Feature data (measurements)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bacterial Foraging parameters
num_clusters = 3
population_size = 20
num_iterations = 50
step_size = 0.1
max_swim_steps = 3  # This is a temporary prototype setting, not a verified thesis parameter.
# These are temporary prototype settings, not verified thesis parameters.
reproduction_interval = 10
reproduction_events = 0
elimination_interval = 10
elimination_probability = 0.10
elimination_events = 0
dispersed_bacteria_count = 0
health_accumulation_steps = 0
# These elimination-dispersal settings are temporary prototype values,
# not verified thesis parameters.

# Get feature bounds for initialization
feature_min = X_scaled.min(axis=0)
feature_max = X_scaled.max(axis=0)
feature_range = feature_max - feature_min


def calculate_fitness(X, centers):
    """
    Calculate the sum of squared Euclidean distances
    between every sample and its assigned cluster center.
    """
    distances = np.linalg.norm(
        X[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2,
    )
    labels = np.argmin(distances, axis=1)
    fitness = np.sum((X - centers[labels]) ** 2)
    return fitness, labels


# Initialize bacteria (each bacterium is a set of cluster centers)
# Each bacterium has shape (num_clusters, num_features)
bacteria = rng.uniform(
    low=feature_min,
    high=feature_max,
    size=(population_size, num_clusters, X_scaled.shape[1])
)

# Calculate initial fitness for all bacteria
fitness_values = np.zeros(population_size)
for i in range(population_size):
    fitness_values[i], _ = calculate_fitness(X_scaled, bacteria[i])

# Health accumulates each bacterium's fitness across one reproduction interval.
# Lower accumulated health is better because lower fitness is better.
health_values = np.zeros(population_size)
accepted_movements = 0

# Main iteration loop
for iteration in range(1, num_iterations + 1):
    # For each bacterium, attempt tumble-and-swim movement
    for i in range(population_size):
        direction = rng.normal(size=bacteria[i].shape)
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            continue

        direction = direction / direction_norm

        for _ in range(max_swim_steps):
            candidate = bacteria[i] + step_size * direction
            candidate_fitness, _ = calculate_fitness(X_scaled, candidate)

            if candidate_fitness < fitness_values[i]:
                bacteria[i] = candidate
                fitness_values[i] = candidate_fitness
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
        # Start a new health-accumulation period after reproduction.
        health_values = np.zeros(population_size)

        reproduction_events += 1

    if iteration % elimination_interval == 0:
        protected_best_index = np.argmin(fitness_values)
        for i in range(population_size):
            if i == protected_best_index:
                continue

            if rng.random() < elimination_probability:
                bacteria[i] = rng.uniform(
                    low=feature_min,
                    high=feature_max,
                    size=(num_clusters, X_scaled.shape[1]),
                )
                fitness_values[i], _ = calculate_fitness(
                    X_scaled,
                    bacteria[i],
                )
                dispersed_bacteria_count += 1

        elimination_events += 1

    # Identify best bacterium after this iteration
    best_idx = np.argmin(fitness_values)
    best_fitness = fitness_values[best_idx]

    # Print progress at specified iterations
    if iteration in [1, 10, 20, 30, 40, 50]:
        print(f"Iteration {iteration}: best fitness = {best_fitness:.4f}")

# Get final best bacterium and its fitness
best_idx = np.argmin(fitness_values)
best_bacterium = bacteria[best_idx]
best_fitness, best_labels = calculate_fitness(X_scaled, best_bacterium)

# Calculate silhouette score
silhouette = silhouette_score(X_scaled, best_labels)

# Count samples in each cluster
unique_labels, counts = np.unique(best_labels, return_counts=True)

# Print final results
print()
print("Final BF Results:")
print(f"Final BF fitness: {best_fitness:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")
print("Cluster sample counts:")
for label, count in zip(unique_labels, counts):
    print(f"  Cluster {label}: {count} samples")
print(f"Total accepted movements: {accepted_movements}")
print(f"Health accumulation steps: {health_accumulation_steps}")
print(f"Reproduction events: {reproduction_events}")
print(f"Elimination-dispersal events: {elimination_events}")
print(f"Total dispersed bacteria: {dispersed_bacteria_count}")
print(f"Final population size: {len(bacteria)}")
