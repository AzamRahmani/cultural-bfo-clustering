# BF Lifecycle and Bacterial Interaction Audit

## 1. Audit scope

This audit compares the current BF and CBF implementations with the lifecycle and bacterial-interaction evidence recorded in [THESIS_EVIDENCE_AUDIT.md](THESIS_EVIDENCE_AUDIT.md) and the current implementation status in [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md). It examines:

- population initialization
- objective evaluation
- tumble
- swim
- greedy acceptance
- bacterial interaction or swarming term Jcc
- bacterial health
- reproduction
- elimination-dispersal
- best-solution preservation
- loop nesting
- stopping conditions

This is a documentation-only comparison. Code behavior is reported as observed; unresolved thesis details remain unresolved. In particular, the evidence audit reports ordinary Euclidean intra-cluster distance on thesis page 66, while current optimization code uses squared Euclidean distance. Thesis pages 54-57 are recorded as describing chemotaxis, tumble, swim, swarming, bacterial health, reproduction, elimination-dispersal, and retained best solution; exact coefficients, loop nesting, and stopping rules still require verification.

## 2. Current reusable BF execution flow

The reusable implementation is `run_bf` in [bf_reusable.py](bf_reusable.py#L43). Line references below are approximate and refer to the current file. Let `n` be samples, `d` features, `k` requested clusters, and `p` population size. A bacterium is a `(k, d)` center matrix; this is the prototype representation, not a settled thesis representation.

| Stage and approximate location | Inputs and shapes | Operation and updated result | Counters changed | Regression protection in `test_project.py` |
|---|---|---|---|---|
| Input checks and initialization, `run_bf` around lines 43-105 | `X_scaled` `(n,d)`; required `num_clusters` `k`; population generated `(p,k,d)`; fitness `(p,)`; health initialized `(p,)` | Validates 2-D data, `k >= 2`, `k <= n`, and `p >= 2`. Uses per-feature minima/maxima and seeded uniform sampling for centers. Calls `calculate_fitness` once per initial bacterium. Raises if all initial candidates are invalid. | Initializes accepted-movement, health-step, reproduction, elimination, dispersed-bacterium, and empty-cluster-rejection counters to zero. No evaluation-count counter exists. | Partly: shapes, finite fitness, and expected cluster count are asserted for historical/reusable runs (roughly lines 171-219). The exact uniform initialization distribution and initial objective computation are not independently tested. |
| Objective evaluation, `calculate_fitness` around lines 7-40 | `X_scaled` `(n,d)` and centers `(k,d)`; pairwise distance array `(n,k)`; labels `(n,)` | Assigns each sample to its nearest center using ordinary Euclidean distance to choose labels, then sums squared coordinate differences for fitness. By default, any empty cluster returns infinite fitness. | No direct counter; callers count rejected invalid movement or dispersal candidates. | Metric helpers in [clustering_metrics.py](clustering_metrics.py#L45) have direct tests for squared and ordinary Euclidean sums (roughly `test_project.py` lines 98-155). Those tests do not directly prove that `run_bf` calls the squared formula or tests the empty-cluster branch; fixed-seed output references cover resulting behavior indirectly. |
| Outer iteration and tumble, around lines 116-125 | One bacterium `(k,d)` at a time; population `(p,k,d)` | For each iteration and bacterium, draws a Gaussian direction `(k,d)`, normalizes it, and skips the bacterium if its norm is zero. This is the implemented random tumble direction; there is no distinct named chemotaxis routine. | None for a successful direction draw or zero-direction skip. | Not isolated: no assertion checks the direction distribution, normalization, or zero-direction handling. Fixed-seed outputs cover the whole run only. |
| Swim and candidate evaluation, around lines 126-136 | Current bacterium and direction `(k,d)`; candidate `(k,d)`; fitness scalar | Reuses the same normalized direction for up to `max_swim_steps` additive moves of `step_size`; each candidate is evaluated. An invalid candidate is rejected and the swim stops. | `empty_cluster_rejections` increments on an invalid candidate. | Not isolated: there is no focused assertion for the number or trajectory of swim attempts. Aggregate output and empty-cluster counts are covered for selected fixed-seed datasets. |
| Greedy acceptance, around lines 137-142 | Candidate fitness and current `fitness_values[index]` | Accepts only a finite candidate with strictly lower fitness; updates that bacterium and its fitness, then tries another swim step. The first non-improving candidate ends that bacterium's swim. | `accepted_movements` increments on acceptance. | Indirect only: no direct assertion exercises better, equal, worse, or invalid candidates. The fixed-seed fitness outputs are the broad regression reference; the movement counter itself is not asserted in `test_project.py`. |
| Health accumulation, around lines 144-145 | `fitness_values` `(p,)`; `health_values` `(p,)` | Adds each bacterium's current fitness once per outer iteration. Lower accumulated health is preferred later. | `health_accumulation_steps` increments once per iteration. | Partly: health array shape and expected 50 accumulation steps are asserted. The accumulated values, reset semantics, and exact relation to reproduction selection are not directly checked. |
| Reproduction, around lines 147-164 | Health and fitness `(p,)`; bacteria `(p,k,d)` | At each `reproduction_interval`, sorts by accumulated health, selects `floor(p/2)`, duplicates selected bacteria and their current fitness to rebuild the population, and resets health to zero. | `reproduction_events` increments once per triggered interval. | Partly: event count (5 at defaults), final population length, and health-step count are checked. Selection ordering, duplication identity, reset values, and timing within the iteration are not independently tested. |
| Elimination-dispersal, around lines 166-202 | Current bacteria `(p,k,d)` and fitness `(p,)`; replacement `(k,d)` | At each `elimination_interval`, identifies the current population's best bacterium and skips it. Each other bacterium is considered for replacement with `elimination_probability`; replacement sampling retries up to 100 times for a valid candidate, then updates that bacterium and its fitness. | `empty_cluster_rejections` increments for invalid replacement attempts; `dispersed_bacteria_count` increments after a replacement; `elimination_events` increments once per triggered interval. | Partly: event counts, some rejection totals, and reproducible result references are checked. The probability model, best-member protection at each event, retry limit, and state transition are not isolated by tests. |
| Final selection and return, around lines 204-229 | Final population `(p,k,d)` and fitness `(p,)` | Returns the minimum-fitness bacterium in the final population, its labels and silhouette, final arrays, parameters represented by the seed, and counters. It does not maintain a separate run-wide archive; protecting the current best at one elimination event does not guarantee preservation across later events. | No additional lifecycle counter. | Partly: returned array shapes, finite metrics, fixed-seed fitness, and cluster counts are asserted. A regression does not establish global-best preservation semantics. |
| Stopping, outer loop around line 116 | `num_iterations` | Runs the fixed `range(1, num_iterations + 1)` loop; there is no convergence or fitness-based early stop. | Health steps and interval-triggered events reflect completed iterations. | Indirectly: default 50 health steps are asserted. There is no focused test for alternate iteration limits or convergence behavior. |

## 3. Historical Iris BF execution flow

[bf_iris.py](bf_iris.py#L1) is a standalone Iris script. It standardizes the Iris features, obtains a configurable seed from `CLUSTER_SEED` (default 42), and uses a population of 20 center matrices with shape `(20, 3, 4)`. `calculate_fitness` around line 51 assigns nearest centers and returns summed squared Euclidean error, without the reusable implementation's default empty-cluster rejection. Initial fitness is evaluated for every bacterium.

Its main loop around lines 84-148 runs 50 iterations. For each bacterium it draws and normalizes a Gaussian direction, then attempts up to three fixed-size moves. Strict improvement is accepted and the same direction is reused; the first non-improvement ends the swim. It accumulates current fitness as health every iteration, duplicates the healthier half every 10 iterations and clears health, and performs elimination-dispersal every 10 iterations with probability 0.10. At an elimination event it protects only the currently best bacterium, samples a replacement once, and accepts that result without retrying for nonempty clusters. The final best is selected from the final population. There is no separate archived global best, no Jcc term, and no separate chemotaxis/reproduction/elimination loop nest.

The historical lifecycle is therefore similar to reusable BF in its simplified tumble/swim, health, reproduction, interval-based elimination-dispersal, and final-population selection. It differs materially in validity handling: historical movement fitness has no empty-cluster rejection, and historical dispersal does not retry invalid replacements. Reusable BF rejects empty-cluster candidates and retries dispersal sampling up to 100 times. Reusable BF also adds reusable input validation, result dictionaries, and reported rejection counters; its core loop remains a prototype rather than a thesis-verified procedure. Neither implementation has a run-wide archived best solution.

## 4. Thesis lifecycle comparison

Status is classified using only the requested categories. For implementation rows, “Verified software behavior” means the described software behavior is visible in the code; it does not certify thesis fidelity.

| Lifecycle element | Thesis evidence | Historical Iris BF | Reusable BF | Status | Scientific risk | Required action |
|---|---|---|---|---|---|---|
| Initialization | Pages 54-57 describe BF lifecycle; candidate encoding details remain ambiguous in the evidence audit. | Uniform `(20,3,4)` center population bounded by standardized Iris feature minima/maxima. | Uniform `(p,k,d)` center population bounded by input feature minima/maxima; default `p=20`. | Requires thesis verification | A center matrix and its initialization distribution may not match the thesis bacterium encoding or population rule. | Verify candidate representation, population size, and initialization rule against the thesis. |
| Objective function | Page 66 evidence states a sum of ordinary Euclidean distances. | Nearest-center labels; fitness is summed squared coordinate error. | Same squared-error objective in `calculate_fitness`; empty-cluster candidates are additionally invalid by default. | Verified software behavior | The optimization objective differs in form from the thesis evidence; optimization and reported metric can diverge. | Verify the thesis equation and intended optimization objective before any algorithm change. |
| Chemotaxis | Chemotaxis is listed as part of the lifecycle; exact nesting and procedural details are not verified. | One outer iteration loop with bacteria traversal and movement. | Same outer iteration plus population traversal; no separate chemotaxis function or loop. | Requires thesis verification | A simplified movement loop can omit or reorder thesis lifecycle work. | Verify the described chemotaxis sequence and loop nesting. |
| Tumble | Tumble is listed in pages 54-57; exact operator and coefficients are unverified. | Normalized Gaussian random direction over `(k,d)`. | Same normalized Gaussian random direction over each bacterium. | Requires thesis verification | Direction distribution, scaling, and update semantics may not match the thesis. | Confirm the tumble operator and step rule from the thesis. |
| Swim | Swim is listed in pages 54-57; swim length and stopping rule are unverified. | Up to 3 moves in the same direction; stop at first non-improvement. | Same behavior with default `max_swim_steps=3`. | Requires thesis verification | Swim length and termination directly affect search and reported outcomes. | Verify swim length and movement sequence. |
| Greedy acceptance | Exact acceptance rule is not specified in the current evidence audit. | Strictly lower fitness is accepted; otherwise swim stops. | Same strict-improvement rule; invalid fitness is rejected first. | Requires thesis verification | A different acceptance or equal-fitness rule changes the search trajectory. | Confirm acceptance semantics in the thesis. |
| Jcc interaction/swarming | Page 55 records an attraction-repulsion interaction term Jcc; no coefficients are established. | No interaction/swarming calculation found. | No interaction/swarming calculation found. | Missing implementation | If required in the implemented thesis variant, its omission changes the BF search dynamics; its equation must not be guessed. | Determine whether the thesis variant requires Jcc and verify its exact equation and coefficients before implementation. |
| Bacterial health | Bacterial health is listed in pages 54-57; exact definition and accumulation ordering are unverified. | Adds current fitness once per iteration; lower health ranks for reproduction; resets after reproduction. | Same accumulation and reset semantics. | Requires thesis verification | Health ranking and accumulation window determine which bacteria reproduce. | Verify health definition, ranking direction, and accumulation window. |
| Reproduction | Reproduction is listed; the evidence audit records reproduction interval 30 on page 67, with remaining mechanics requiring verification. | Every 10 iterations, duplicates the health-ranked better half. | Same default interval 10 and half-population duplication. | Temporary prototype setting | Default timing conflicts with page-67 interval evidence; selection and duplication could also differ. | Verify interval and selection/duplication procedure. |
| Elimination-dispersal | Listed in lifecycle evidence; exact timing, count, and probability require verification. | Every 10 iterations, independently samples non-best bacteria at probability 0.10; one replacement attempt. | Same interval/probability defaults, but retries up to 100 times for valid replacement. | Temporary prototype setting | Timing, replacement validity policy, and probability alter population diversity and outcomes. | Verify timing, probability, number of bacteria affected, and replacement policy. |
| Archived best solution | Thesis pages 54-57 include retained best solution. | No persistent archive; final result is selected from final population. | No persistent BF archive; protects only the current best at each dispersal event and returns the final population best. | Missing implementation | A good solution may be lost across later iterations or elimination events. | Verify the thesis preservation rule; implement only in a separate thesis-compatible implementation if required. |
| Loop nesting | Lifecycle elements are present in evidence, but exact nesting is explicitly unverified. | One iteration loop encloses bacterium movement; health/reproduction/elimination logic follows each population pass. | Same simplified single iteration loop; reproduction and elimination are interval conditionals, not separate lifecycle loops. | Temporary prototype setting | Scheduling may not reproduce the thesis sequence or per-chemotaxis event counts. | Verify the complete loop nesting and event order. |
| Stopping conditions | Exact stopping rules require thesis verification. | Fixed 50-iteration bound, with no convergence stop. | Fixed `num_iterations` bound, default 50; no convergence stop. | Temporary prototype setting | A fixed bound can terminate before or after the thesis criterion. | Verify maximum iterations and any convergence/stagnation stopping condition. |

## 5. Jcc interaction audit

The tracked project Python sources were searched for each requested term, case-insensitively: `Jcc`, `interaction`, `attract`, `repel`, `swarming`, `cell-to-cell`, `d_attract`, `w_attract`, `h_repel`, and `w_repel`.

**Result: zero matches for all requested terms in project Python source files.** No Jcc expression, attraction/repulsion coefficient, cell-to-cell calculation, or related variable/function exists in the BF or CBF implementation. The evidence audit states that page 55 contains an attraction-repulsion Jcc term, but explicitly provides no coefficient values. The implementation status is **Missing implementation**. This audit does not infer an equation or coefficients and does not propose a substitute.

## 6. Loop structure audit

The thesis evidence audit records lifecycle elements, but says exact nesting is still unverified. Current historical and reusable BF each have a single bounded iteration loop, a bacteria loop for movement, then conditionals that trigger reproduction and elimination-dispersal on iteration-number intervals. They do not explicitly separate chemotaxis loops, reproduction loops, and elimination-dispersal loops as independent lifecycle nests. `max_swim_steps` provides an inner movement loop, not a thesis-verified chemotaxis schedule.

The reusable defaults use simplified interval triggers (`reproduction_interval=10`, `elimination_interval=10`) and a fixed iteration limit (`num_iterations=50`). Under the required status taxonomy, this structure and schedule are **Temporary prototype setting**. Verify nesting and stopping conditions against the thesis before creating a thesis-compatible implementation.

## 7. Parameter inventory

The following are all arguments accepted by `run_bf` in [bf_reusable.py](bf_reusable.py#L43). `X_scaled` and `num_clusters` are required positional/keyword arguments and have no default. `X_scaled` is expected to be standardized data shaped `(n_samples, n_features)`; `num_clusters` is the requested `k`. The thesis evidence audit lists population size 10, maximum iterations 100, and reproduction interval 30 from page 67. It does not establish the other BF values below; no thesis values are inferred for them.

| `run_bf` parameter | Default | Classification | Evidence / verification note |
|---|---:|---|---|
| `X_scaled` | Required; no default | Requires thesis verification | Input is documented as standardized data. Confirm exact thesis preprocessing and candidate-data requirements. |
| `num_clusters` | Required; no default | Requires thesis verification | Must be provided by the caller; confirm cluster-count rule against the thesis for each dataset. |
| `random_seed` | `42` | Temporary prototype setting | Reproducibility control in the software; the evidence audit does not establish a thesis seed protocol. |
| `population_size` | `20` | Temporary prototype setting | Differs from the page-67 thesis evidence value of 10; confirm source evidence before treating that value as definitive. |
| `num_iterations` | `50` | Temporary prototype setting | Differs from the page-67 maximum-iteration evidence value of 100; exact stopping criterion also needs verification. |
| `step_size` | `0.1` | Requires thesis verification | No thesis value is present in the evidence audit. |
| `max_swim_steps` | `3` | Temporary prototype setting | Historical script explicitly marks this as temporary; no thesis swim length is established in the evidence audit. |
| `reproduction_interval` | `10` | Temporary prototype setting | Differs from the page-67 thesis evidence interval of 30. |
| `elimination_interval` | `10` | Temporary prototype setting | No thesis interval is established in the evidence audit. |
| `elimination_probability` | `0.10` | Temporary prototype setting | No thesis probability is established in the evidence audit. |

The `require_all_clusters` argument belongs to `calculate_fitness`, not `run_bf`, so it is not part of this `run_bf` parameter inventory. `run_cbf` has additional cultural parameters; those are outside the BF inventory.

## 8. CBF inheritance

Reusable CBF does not call `run_bf` or share a BF lifecycle helper: [cbf_reusable.py](cbf_reusable.py#L43) duplicates the fitness and population lifecycle structure. It reuses the same center-matrix population, squared-distance fitness with empty-cluster rejection, normalized random tumble, bounded same-direction swim, strict greedy acceptance, health accumulation, interval-triggered reproduction, and interval-triggered elimination-dispersal with best-current-member protection.

CBF adds a situational `belief_best_bacterium` and `belief_best_fitness` archive, a cultural direction combined with each random tumble direction, and normative coordinate bounds derived from the best `ceil(population_size * acceptance_ratio)` members. Candidate centers are clipped to those bounds. It updates belief and normative state after movement and before health accumulation. The CBF result is the archived belief-best solution, whereas BF returns the best member of the final population; CBF also reports final-population best separately. Elimination-dispersal protects the current population best, not the cultural archive as a population member.

These are implementation differences only. Reusable CBF still duplicates rather than factors out the BF lifecycle, and its shared BF behavior has the same unverified thesis objective, Jcc absence, loop nesting, and prototype schedule. Cultural Algorithm correctness is outside this audit.

## 9. Regression-test coverage

`test_project.py` imports and exercises historical Iris BF/CBF and reusable BF/CBF. For BF lifecycle behavior it currently checks:

- input/output shapes for centers, populations, fitness values, health values, and labels
- finite best fitness and a requested number of nonempty clusters
- selected fixed-seed BF/CBF result values and dataset-specific rejection totals
- default health accumulation count (50), reproduction event count (5), elimination event count (5), and final population size
- CBF archive/belief relationships and normative-bound invariants
- squared- and ordinary-Euclidean metric helper results and validation errors

It does not directly test Jcc, direction generation, individual swim trajectories, greedy-acceptance branches, the actual `run_bf` objective equation, health values or reset ordering, reproduction selection identities, elimination probability/protection/retry semantics, persistent BF best preservation, or thesis loop nesting/stopping criteria. Lifecycle counters and fixed-seed fitness assertions protect selected aggregate software outcomes, not each underlying operation.

**Passing tests protects current software behavior but does not prove thesis fidelity.** The seed-42 values documented by the evidence/status files are regression references, not thesis targets.

## 10. Prioritized findings

**Priority 1**

- Determine whether Jcc is required by the implemented thesis variant; verify its equation and coefficients from the thesis before implementation.
- Verify the objective equation used during optimization. The current reusable and historical BF use squared Euclidean error; the evidence audit records ordinary Euclidean distance on thesis page 66.
- Verify loop nesting and stopping conditions, including whether chemotaxis, reproduction, and elimination-dispersal have distinct loops or schedules.

**Priority 2**

- Verify population size.
- Verify swim length.
- Verify reproduction timing.
- Verify elimination-dispersal timing and probability.
- Verify health accumulation and ranking.

**Priority 3**

- Design any thesis-compatible BF implementation separately, preserving the current prototype and its regression references.
- Add focused lifecycle tests only after the equations, parameters, and event sequence have been verified.

## 11. Safety boundary

- No BF or CBF algorithm changes are authorized by this audit.
- Existing regression references must remain unchanged.
- Missing components must not be added until their equations and parameters are verified.
- A thesis-compatible BF should be implemented separately from the preserved prototype.
- Project B is not part of this milestone.

## 12. Run

Run `git status --short` after creating this document. Expected output for this documentation-only milestone, assuming a clean starting worktree:

```text
?? BF_LIFECYCLE_AUDIT.md
```

## 13. Report

- **Jcc exists?** No. The requested interaction-term search found no matches in tracked project Python sources; classify the attraction-repulsion term as **Missing implementation**.
- **Major BF lifecycle differences:** Historical Iris BF and reusable BF both use simplified single-loop interval scheduling and no run-wide archived best. Reusable BF adds empty-cluster rejection and retries replacement sampling up to 100 attempts; historical BF does neither. Both select the final result from the final population and both use squared-error optimization.
- **Parameters requiring verification:** `X_scaled` preprocessing and `num_clusters`; `step_size`; population size, iteration bound, and reproduction interval versus the recorded thesis values; swim length; elimination interval and probability; and the health definition/schedule. Exact source evidence must be checked before changing prototype defaults.
- **Regression-test gaps:** No direct Jcc, objective-equation, movement-branch, health-value/order, selection-identity, elimination-probability, BF archived-best, or thesis loop/stopping tests. Existing tests cover dimensions, finite outputs, selected fixed-seed results, and some event counts.
- **Files changed by this milestone:** This document only, `BF_LIFECYCLE_AUDIT.md`. No algorithm or regression-reference files are modified.
