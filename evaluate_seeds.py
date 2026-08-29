import os
import re
import subprocess
import sys
import numpy as np

seeds = [0, 1, 2, 3, 4]


def run_experiment(script_name, seed, fitness_label):
    environment = os.environ.copy()
    environment["CLUSTER_SEED"] = str(seed)

    completed_process = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )

    fitness_match = re.search(
        rf"{fitness_label}: ([0-9.]+)",
        completed_process.stdout,
    )

    silhouette_match = re.search(
        r"Silhouette Score: ([0-9.]+)",
        completed_process.stdout,
    )

    if fitness_match is None:
        raise ValueError(
            f"Could not read fitness from {script_name}"
        )

    if silhouette_match is None:
        raise ValueError(
            f"Could not read silhouette from {script_name}"
        )

    fitness = float(fitness_match.group(1))
    silhouette = float(silhouette_match.group(1))

    return fitness, silhouette


bf_fitness_values = []
bf_silhouette_values = []
cbf_fitness_values = []
cbf_silhouette_values = []

for seed in seeds:
    bf_fitness, bf_silhouette = run_experiment(
        "bf_iris.py",
        seed,
        "Final BF fitness",
    )
    bf_fitness_values.append(bf_fitness)
    bf_silhouette_values.append(bf_silhouette)

    cbf_fitness, cbf_silhouette = run_experiment(
        "cbf_iris.py",
        seed,
        "Final CBF fitness",
    )
    cbf_fitness_values.append(cbf_fitness)
    cbf_silhouette_values.append(cbf_silhouette)

    print(f"Seed {seed}")
    print(f"BF fitness: {bf_fitness}")
    print(f"BF silhouette: {bf_silhouette}")
    print(f"CBF fitness: {cbf_fitness}")
    print(f"CBF silhouette: {cbf_silhouette}")
    print()

bf_fitness_values = np.array(bf_fitness_values)
bf_silhouette_values = np.array(bf_silhouette_values)
cbf_fitness_values = np.array(cbf_fitness_values)
cbf_silhouette_values = np.array(cbf_silhouette_values)

bf_fitness_mean = np.mean(bf_fitness_values)
bf_fitness_std = np.std(bf_fitness_values, ddof=0)
bf_silhouette_mean = np.mean(bf_silhouette_values)
bf_silhouette_std = np.std(bf_silhouette_values, ddof=0)

cbf_fitness_mean = np.mean(cbf_fitness_values)
cbf_fitness_std = np.std(cbf_fitness_values, ddof=0)
cbf_silhouette_mean = np.mean(cbf_silhouette_values)
cbf_silhouette_std = np.std(cbf_silhouette_values, ddof=0)

print("Multiple-Seed Summary")
print()
print(f"BF mean fitness: {bf_fitness_mean:.4f}")
print(f"BF fitness standard deviation: {bf_fitness_std:.4f}")
print(f"BF mean silhouette: {bf_silhouette_mean:.4f}")
print(f"BF silhouette standard deviation: {bf_silhouette_std:.4f}")
print()
print(f"CBF mean fitness: {cbf_fitness_mean:.4f}")
print(f"CBF fitness standard deviation: {cbf_fitness_std:.4f}")
print(f"CBF mean silhouette: {cbf_silhouette_mean:.4f}")
print(f"CBF silhouette standard deviation: {cbf_silhouette_std:.4f}")
print()

cbf_fitness_wins = np.sum(
    cbf_fitness_values < bf_fitness_values
)
cbf_silhouette_wins = np.sum(
    cbf_silhouette_values > bf_silhouette_values
)

print(f"CBF lower-fitness wins: {cbf_fitness_wins} of 5")
print(f"CBF higher-silhouette wins: {cbf_silhouette_wins} of 5")
print()

print("Important notes:")
print("- Lower fitness is better.")
print("- Higher silhouette is generally better.")
print("- Results use five independent random seeds.")
print("- This is an initial repeated-seed evaluation.")
print("- Five seeds are still insufficient for strong statistical conclusions.")
print("- The CBF implementation is still an exploratory prototype.")
print()

assert len(bf_fitness_values) == 5
assert len(cbf_fitness_values) == 5

assert np.all(np.isfinite(bf_fitness_values))
assert np.all(np.isfinite(cbf_fitness_values))
assert np.all(np.isfinite(bf_silhouette_values))
assert np.all(np.isfinite(cbf_silhouette_values))

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
