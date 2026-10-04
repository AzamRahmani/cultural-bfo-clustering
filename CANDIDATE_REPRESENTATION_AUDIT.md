# Candidate Representation Audit

## 1. Audit Purpose

This audit determines how a bacterium or chromosome is represented in the current software and records what remains uncertain about the thesis representation. It compares the existing prototype structures with evidence already recorded in the repository audits.

No implementation decision is authorized by this audit.

## 2. Current Historical Iris BF Representation

In `bf_iris.py`, the Iris feature matrix `X_scaled` has shape `(150, 4)`. The prototype uses `num_clusters = 3` and `population_size = 20` (parameter setup near lines 23-43). One bacterium, also called `best_bacterium` when selected, has shape `(3, 4)`: axis 0 indexes the three cluster centers and axis 1 indexes the four standardized features. The complete `bacteria` population has shape `(20, 3, 4)`: axis 0 indexes bacteria, axis 1 centers, and axis 2 features.

The initialization block (around lines 66-74) samples each center coordinate uniformly between that feature's minimum and maximum in `X_scaled`, using `rng.uniform` with size `(population_size, num_clusters, X_scaled.shape[1])`.

`calculate_fitness` (around lines 49-55) computes Euclidean distances from each sample to each center, assigns each sample to its nearest center with `argmin`, and returns one cluster label per sample, shape `(150,)`. Fitness is the sum of squared coordinate differences from each sample to its assigned center. This historical function does not reject candidates with empty clusters.

In the main loop (around lines 84-114), `rng.normal(size=bacteria[i].shape)` creates a tumble direction with the same `(3, 4)` shape as one bacterium; the direction is normalized. Each swim candidate is formed by adding `step_size * direction` to that bacterium. A candidate replaces the current bacterium only when its fitness is strictly lower; the same direction is reused for up to `max_swim_steps` moves, stopping at the first non-improvement. These operations change center coordinates while preserving the `(3, 4)` shape.

The cluster count is fixed at three in this script. Tumble, swim, reproduction, and elimination-dispersal preserve the center-matrix dimensions; no operation in this historical BF script changes the number of clusters.

## 3. Current Reusable BF Representation

`run_bf` in `bf_reusable.py` accepts `X_scaled` with symbolic shape `(n_samples, n_features)` and a requested `num_clusters`, denoted `n_clusters` here. One bacterium is a center matrix of shape `(n_clusters, n_features)`. The population has shape `(population_size, n_clusters, n_features)`, with axes for bacteria, centers, and features respectively. The returned labels have shape `(n_samples,)`.

`run_bf` validates the data and requested cluster count, then initializes each center coordinate uniformly within the corresponding feature's observed minimum and maximum (initialization around lines 43-105). Each candidate's rows are interpreted as centers. `calculate_fitness` (around lines 7-40) computes sample-to-center distances and assigns each sample to the nearest center with `argmin`. Fitness is the sum of squared residual coordinates to the assigned centers. By default, if fewer than `n_clusters` centers receive a sample, `calculate_fitness` returns infinite fitness; movement and dispersal callers reject such candidates.

During the movement loop (around lines 116-142), the tumble direction is sampled with the same shape as one bacterium, `(n_clusters, n_features)`, and normalized. A swim step adds `step_size * direction` to the center matrix. Strictly lower finite fitness is accepted; the same direction is used for subsequent swim steps up to `max_swim_steps`, and a non-improving or invalid candidate stops that swim.

At reproduction (around lines 147-164), bacteria are ranked by accumulated health. The `population_size // 2` healthiest (lowest-health) bacteria are selected and copied twice along the population axis, together with their fitness values; health is reset. The resulting population size is `2 * floor(population_size / 2)` (equal to `population_size` for the default even population size).

At elimination-dispersal (around lines 166-202), the current lowest-fitness bacterium is protected. Other bacteria may be replaced according to the configured probability. A replacement is a newly sampled center matrix of shape `(n_clusters, n_features)`; up to 100 samples are attempted until one has finite fitness. Neither reproduction nor elimination-dispersal changes `n_clusters` or the shape of one bacterium.

## 4. Current Reusable CBF Representation

`run_cbf` in `cbf_reusable.py` uses the same center-matrix representation and principal BF rules as `run_bf`: `X_scaled` has shape `(n_samples, n_features)`, one bacterium has shape `(n_clusters, n_features)`, the population has shape `(population_size, n_clusters, n_features)`, and labels have shape `(n_samples,)`. It initializes feature-bounded center matrices, assigns samples to nearest centers, calculates squared residual fitness, rejects empty-cluster candidates, and uses same-shaped tumble directions and additive swim updates. It also uses health-based reproduction and same-shaped elimination replacements. The implementation duplicates these rules rather than calling `run_bf` or sharing its lifecycle helper (see `run_cbf`, beginning around line 43).

CBF-specific state also uses the center-matrix shape:

- `belief_best_bacterium` is a copy of the best current initial or improved bacterium, shape `(n_clusters, n_features)`. It is the prototype's situational-knowledge archive.
- Situational knowledge influences movement by subtracting the current bacterium from `belief_best_bacterium`, normalizing that matrix-shaped direction, scaling it by `cultural_influence`, and combining it with the random tumble direction. The combined direction is normalized before the swim update. This moves center coordinates in the same representation; it does not change cluster count.
- `normative_lower` and `normative_upper` each have shape `(n_clusters, n_features)`. They are computed by element-wise minima and maxima over the selected better bacteria, and updated from the current accepted population. Candidate center coordinates are clipped element-wise to these bounds before fitness evaluation.
- The acceptance count selects a number of whole center matrices based on `population_size` and `acceptance_ratio`; it does not change the matrix dimensions.

This section documents the prototype's representation and operations only. It does not evaluate whether the implementation is a correct or complete Cultural Algorithm.

## 5. Thesis Evidence

The thesis evidence recorded in `THESIS_EVIDENCE_AUDIT.md` describes bacteria as positions in a search space and, on pages 60-62, describes genes as encoding cluster assignments or cluster-related information. That audit also records replacement, division/splitting, merging, repair, and gene transfer as structural operations, along with a maximum-cluster constraint.

The evidence audit states that the exact encoding remains ambiguous and appears to be more than a simple center matrix. It does not establish the exact gene layout, how a gene maps to a cluster or feature, or the detailed mutations performed by each structural operation. In particular, the audit records replacement as an operation but does not specify here exactly what replacement changes.

The center-matrix representation is therefore verified as current software behavior, not as a confirmed thesis representation. The available thesis evidence does not authorize replacing it, converting it to assignment genes, or changing the cluster-count behavior.
