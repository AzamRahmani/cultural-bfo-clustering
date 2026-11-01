# Experiment V1 Run Record

This record documents the single predefined execution of the 240-run
prototype experiment defined in `EXPERIMENT_PROTOCOL.md`. It reports what
happened. It does not analyze or compare algorithm results.

## 1. Execution

| Item | Value |
|---|---|
| Results | `results/prototype_30_seed_results.csv` |
| Run metadata | `results/prototype_30_seed_results.metadata.json` |
| Code commit | `3f29075540e0680d783b89279c75275278970d8a` |
| Working tree | clean (`git_tree_clean: true`, `allow_dirty_tree: false`) |
| Command | `.venv\Scripts\python.exe run_prototype_experiments.py --output results/prototype_30_seed_results.csv` |
| Started (UTC) | 2026-11-01T23:48:17+00:00 |
| Completed (UTC) | 2026-11-01T23:50:37+00:00 |
| Environment | Windows-11-10.0.26300-SP0, Python 3.14.5, NumPy 2.5.2, scikit-learn 1.9.0, SciPy 1.18.0, Matplotlib 3.11.1 |
| CSV SHA-256 | `68d407c7bb9e4824383b1122c161c0925eb11a38f867460f33abcfa88d622eb8` |

The experiment was executed once. No run was repeated, replaced, or removed.

## 2. Output Validation (Protocol Section 15)

All checks passed.

| Check | Result |
|---|---|
| Exactly 240 records, failures retained | 240 records, 232 successful, 8 failed |
| Unique dataset-algorithm-seed keys | passed |
| Seeds 0 through 29 for every dataset-algorithm pair | passed |
| Finite metrics for every successful run | passed |
| Nonempty cluster count equals requested count for every successful run | passed |
| Recorded `num_clusters` and `random_seed` match each run | passed |
| `correct_count + incorrect_count` equals the dataset sample count | passed |
| BF `final_population_best_fitness` empty; CBF `best_fitness <= final_population_best_fitness` | passed |
| Sidecar fingerprint, row counts, commit, and timestamps consistent | passed |

Two Section 15 requirements are enforced by the runner rather than read back
from the CSV, and are covered by `test_project.py`:

- Known labels are passed only to `clustering_error_rate` after each run; the
  optimizers receive `X_scaled` only.
- For CBF, returned `centers` must exactly equal `belief_best_bacterium`;
  otherwise the run is recorded as failed.

## 3. Failed Runs

All failures occurred on Glass (6 requested clusters).

| Dataset | Algorithm | Seed | Recorded failure |
|---|---|---:|---|
| glass | BF | 13 | RuntimeError: The initial population contains no valid clustering solution. |
| glass | BF | 19 | RuntimeError: The initial population contains no valid clustering solution. |
| glass | BF | 28 | RuntimeError: The initial population contains no valid clustering solution. |
| glass | CBF | 9 | ValueError: CBF best_fitness exceeds final_population_best_fitness. |
| glass | CBF | 11 | ValueError: CBF best_fitness exceeds final_population_best_fitness. |
| glass | CBF | 13 | RuntimeError: The initial population contains no valid clustering solution. |
| glass | CBF | 19 | RuntimeError: The initial population contains no valid clustering solution. |
| glass | CBF | 28 | RuntimeError: The initial population contains no valid clustering solution. |

### 3.1 Invalid initial population

**Verified software behavior.** For seeds 13, 19, and 28, every one of the 20
randomly initialized Glass candidates left at least one of the 6 clusters
empty. Under the temporary empty-cluster policy every candidate therefore had
infinite fitness, and both reusable implementations stop with this error
(`bf_reusable.py` and `cbf_reusable.py`, initial-population check). BF and
CBF fail on the same seeds because both draw the initial population from the
same seeded random-number sequence.

**Temporary prototype setting.** This failure mode is a consequence of the
temporary empty-cluster policy and of uniform initialization within feature
bounds. It is not evidence about the thesis method.

### 3.2 CBF archive does not observe the final dispersal

**Verified software behavior.** In `cbf_reusable.py`, the situational
archive (`belief_best_bacterium`, `belief_best_fitness`) is updated after
movement and before reproduction and elimination-dispersal in each
iteration. With 50 iterations and an elimination interval of 10, the final
iteration (50) ends with an elimination-dispersal event. Newly dispersed
random bacteria created in that last event are never compared with the
archive. If one of them is better than the archived solution, the final
population contains a solution better than the returned archived result.

Re-running the two failed configurations outside the repository reproduces
the gap exactly:

| Run | Archived `best_fitness` | `final_population_best_fitness` |
|---|---:|---:|
| glass CBF seed 9 | 5745.5554 | 4566.9490 |
| glass CBF seed 11 | 5433.7670 | 5323.8876 |

The runner's archived-best consistency check (Protocol Section 15) detected
both cases and recorded them as failures instead of reporting an
inconsistent success.

**Consequences.**

- For CBF, `best_fitness` is the archived best of iterations 1 through 50
  before the final dispersal, not the best solution ever evaluated.
- In the 115 successful CBF runs the check passed, so the returned archived
  solution was at least as good as the final population.
- BF has no persistent archive and returns the best solution in the final
  population (see `BF_LIFECYCLE_AUDIT.md`). The two methods therefore select
  their returned result differently; this asymmetry must be stated whenever
  BF and CBF `best_fitness` are compared.

**Not repaired.** Protocol Section 13 forbids repairing an algorithm after
seeing experimental results under this protocol. The defect is documented
here and left unchanged. A repair requires a new protocol version and a new
experiment.

## 4. Paired Completeness

BF and CBF share the seed list, so comparisons are paired by seed.

| Dataset | Seeds with both BF and CBF successful |
|---|---:|
| Iris | 30 |
| Wine | 30 |
| Glass | 25 (seeds 9, 11, 13, 19, 28 excluded) |
| Cancer | 30 |

The handling of incomplete pairs is fixed in `ANALYSIS_PLAN.md` before any
metric is summarized.
