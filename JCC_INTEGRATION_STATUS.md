# Jcc Integration Status

## 1. Current Verified Evidence

The thesis evidence audit records an attraction-repulsion interaction term, Jcc, on page 55. The standalone utility in `bacterial_interaction.py` implements that formula:

```text
squared_distance_i = sum((candidate - population[i]) ** 2)
Jcc = sum_i(
    -attraction_depth * exp(-attraction_width * squared_distance_i)
    + repulsion_height * exp(-repulsion_width * squared_distance_i)
)
```

This utility is standalone. It is not called by `bf_iris.py`, `bf_reusable.py`, `cbf_iris.py`, or `cbf_reusable.py`. The current BF and CBF execution paths therefore do not include this interaction term. Importing and testing the utility in `test_project.py` does not connect it to either algorithm.

## 2. Missing Coefficient Values

The following coefficients are required by the equation:

- `attraction_depth`
- `attraction_width`
- `repulsion_height`
- `repulsion_width`

No numerical values for these coefficients have yet been verified from the thesis. No defaults are proposed here.

## 3. Self-Interaction Ambiguity

The thesis Jcc sum appears to include all bacteria, but the available evidence does not explicitly clarify whether the evaluated bacterium interacts with itself.

When the candidate position equals its own population position, its squared distance is zero. Including that self-interaction contributes:

```text
-attraction_depth + repulsion_height
```

The standalone utility supports explicit inclusion or index-based exclusion of one population row. This is a utility capability, not a resolution of the thesis ambiguity. No self-interaction choice is authorized for thesis-compatible BF yet.

## 4. Candidate Representation Dependency

Jcc operates on bacterial positions. The current reusable BF and CBF prototypes represent each bacterium as a center matrix with shape `(number_of_clusters, number_of_features)`; their population has shape `(population_size, number_of_clusters, number_of_features)`. The standalone utility instead accepts a position vector with shape `(number_of_dimensions,)` and a population matrix with shape `(number_of_bacteria, number_of_dimensions)`.

The available thesis evidence does not settle the bacterial encoding, and no mapping from the prototype's center matrices to Jcc position vectors has been verified. The center-matrix population therefore cannot be passed directly to the standalone utility under its current shape contract. A thesis-supported position representation and corresponding mapping must be established before integration is specified.
