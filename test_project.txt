import io
import numpy as np
from contextlib import redirect_stdout


with redirect_stdout(io.StringIO()):
    import iris_baseline
    import bf_iris
    import cbf_iris


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

print("Project checks passed.")
print(f"K-Means fitness: {iris_baseline.inertia:.4f}")
print(f"BF fitness: {bf_iris.best_fitness:.4f}")
print(f"CBF fitness: {cbf_iris.best_fitness:.4f}")
print("All methods produced three nonempty clusters.")
