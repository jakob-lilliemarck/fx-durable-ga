## Active Work

---

### F1 — Evaluation Noise Estimation

A standalone diagnostic service. Shares only `genotype_evaluation_service` and
the budget aggregate as dependencies. Can be run before or alongside the GA.

#### Responsibility
Estimate the noise floor of the evaluation function — variance in fitness when
evaluating the same genotype repeatedly. Expressed as mean and standard
deviation. All skip decisions in Q1 are expressed relative to this baseline.

#### Design
- The framework randomly generates a small population of probe genotypes
- Each probe is evaluated repeatedly via `genotype_evaluation_service`,
  debiting a dedicated F1 budget on each evaluation
- Stopping criterion per probe: rolling std dev has changed by less than
  epsilon over successive estimates **and** `min_evaluations_per_probe` has
  been reached — guards against false convergence by chance
- Hard stop when budget is exhausted regardless of convergence
- If the evaluation function is deterministic, std dev converges to zero
  immediately after `min_evaluations_per_probe` — F1 exits cheaply and reports
  a noise floor of zero, which is the correct and useful answer
- If noise floors differ meaningfully across probes, report **per fitness-band
  noise floors** rather than a single global value

#### User-configurable parameters
- `probe_population_size`
- `min_evaluations_per_probe`
- `evaluations_per_probe` (budget per probe, soft target)
- `epsilon` (convergence threshold)
- `budget` (hard cap on total evaluations)

#### Aggregate & persistence
Lives in the **genotype aggregate** — noise is a derived fact about a
genotype's evaluation history.

New table: `noise_estimates`
- `genotype_id`
- `mean`, `std_dev`   — current estimate
- `sample_count`      — evaluations contributed
- `converged`         — stopping criterion met, or budget exhausted
- `samples`           — individual fitness values for stabilization chart

#### Output
- Stabilization chart (sample count vs. rolling std dev) per probe
- Global and per fitness-band noise floor estimates

---

### F2 — Embedding Space Quality Diagnostics

Passive diagnostics running continuously alongside the GA. No user intervention
required once configured.

#### Responsibility
Measure whether proximity in embedding space predicts similarity in fitness —
i.e. whether the encoder has sufficient resolving power to support
distance-based decisions.

#### Key insight
Apparent smoothness is not actual smoothness. A low-dimensional encoder may
compress jagged regions into tight neighborhoods, making the landscape look
safe to skip across when it isn't. F2 surfaces this risk.

#### Event trigger
`genotype_indexing_service` re-emits `GenotypeEmbeddingCreatedEvent` when a
new genotype embedding is processed. F2 listens for this event. By the time it
fires, both fitness and embedding are guaranteed to exist — no additional
outbox needed.

---

#### F2a — Fitness-Distance Correlation

**What it measures:** whether large embedding distance predicts large fitness
delta. Weak correlation means no threshold D will produce reliable skips.

**Computation:**
- On each `GenotypeEmbeddingCreatedEvent`, add genotype to a rolling window of
  N most-recently indexed genotypes (N user-configurable)
- Compute pairwise embedding distances and fitness deltas across the window —
  O(N²), negligible for reasonable N since both values are already in db
- Compute **Spearman's correlation** (preferred over Pearson — relationship not
  expected to be linear)
- Persist coefficient as a float on every update — gives a full time-series
  for free at negligible storage cost

**Composition:**
`genotype_embedding_quality_diagnostic_service` coordinates:
- `embedding_repository` → pairwise distances for the window
- `genotype_repository` → fitness values for the window

Correlation computed in application code. Neither repository knows of the other.

**Aggregate & persistence:** `fitness_distance_correlation`
- `correlation_window` — rolling window of (genotype_id, embedding_id, fitness)
  tuples, evicted as window slides
- `correlation_series` — time-series of Spearman coefficients, one float per
  update, annotated with generation marker

**Output:**
- Live rolling scatter plot of (embedding distance, fitness delta) pairs
- Time-series of Spearman coefficient, annotated with generation boundaries

---

#### F2b — Neighborhood Fitness Variance

**What it measures:** fitness roughness at the encoder's current resolution,
locally around each genotype. High variance in a tight neighborhood means the
encoder is collapsing meaningfully different genotypes into the same region —
unresolved roughness. This is a three-way interaction: actual landscape
structure, encoder resolving power, and chosen k.

**Computation (on-demand):**
Three-way composition in the service layer:
1. `genotype_indexing_service` → resolves embedding IDs from genotype IDs via tags
2. `embedding_repository` → k nearest neighbor embedding IDs per embedding
   (`neighbor_index <= k`, not just the kth)
3. `genotype_repository` → fitness values for neighbor genotype IDs

Fitness variance computed in application code. Run for multiple values of k
simultaneously — variance low at k=3 but high at k=10 indicates locally smooth
but regionally rough landscape, which is itself diagnostic.

On-demand for now — straightforward to convert to a continuous computation
later as a performance optimization once the concept is validated.

**No new aggregate** — purely a service-layer computation over existing data.

**Output:**
- Per-genotype neighborhood fitness variance
- Distribution of variance across a population
- Variance plotted for multiple k values simultaneously

---

### Q1 — Similarity-Based Evaluation Skipping

> *Can the framework identify candidates close enough to already-evaluated
> neighbors that evaluation can be skipped, and provide audit tooling to
> measure the cost of that decision?*

**Framework:** compute nearest-neighbor distance before evaluation; skip
candidates below user-defined threshold D; evaluate a random audit sample
(default 5–10%) to measure skip error as |predicted − actual fitness| in
noise-floor units from F1.

**User controls:** threshold D, audit sample rate, skipping on/off.

**Success metrics:**
- **Skip rate** — fraction of evaluations avoided
- **Skip error distribution** — in noise-floor units; error at or below the
  floor means skipping works as well as measurement allows; error above it
  implicates unresolved roughness, pointing the user to F2
- **Convergence impact** — solution quality vs. evaluations consumed, compared
  against a full-evaluation baseline

---

## Architecture

### Service Map
```
budget_repository                        (new, foundation)
        ↑
genotype_evaluation_service              (new, core)
        ↑
        ├── ga_service                   (existing, refactored)
        └── genotype_evaluation_noise_diagnostic_service   (F1, new)

genotype_indexing_service                (existing, renamed)
        ↑
        └── genotype_embedding_quality_diagnostic_service  (F2, new)
                ├── embedding_repository
                └── genotype_repository
```

---

### Aggregate Map
```
budget aggregate                         (new, foundation)
  └── budget_transactions

genotype aggregate
  ├── genotypes                          (existing)
  ├── evaluations                        (existing)
  └── noise_estimates                    (new, F1)

embedding aggregate
  ├── embeddings                         (existing)
  └── outbox                             (existing)

fitness_distance_correlation aggregate   (new, F2a)
  ├── correlation_window
  └── correlation_series
```

---

## Priority Checklist

Prio 1
- [x] Verify indexer digest is deterministic — `Registry::get_indexer_id` returns the same digest for repeated calls with the same indexer (test passes)
- [x] Add tests for `budgeting::Service` — `src/services/budgeting/service.rs`
- [x] Add tests for `locking::Service` — `src/services/locking/service.rs`
- [ ] Add tests for `evaluation::Service` — `src/services/evaluation/service.rs`
- [ ] Add tests for `genotype_explorer::Service` — `src/services/genotype_explorer/service.rs`
- [ ] Fill missing tests for `indexing::Service` — `src/services/indexing/service.rs`
- [ ] Add tests for `optimization::Service` — `src/services/optimization/service.rs`
- [ ] Fill missing tests for `genotype_indexing::Service` — `src/services/genotype_indexing/service.rs`
- [ ] Add visualization of indexes to the app at "src/bin/lineage/main.rs". I am unsure exactly how as of now, but we should be able to visualize k-nearest neighbours for a particular indexer id and request. We should also display the genome as json. It does not have to fit into the same view as lineage - the app may have more than one page.

In more detail:
in the lineage view:
1. Mark the "selected genotype" with an accent color
2. Provide a hover-over showing some data about the each genotype including:
  a. ID
  b. Fitness
  c. Genome (as well formatted json)
3. Contain the SVG area, and on right side, show a list of N similar genotypes as the currently selected one (later on we can introduce pagination, for now just N most similar).
4. Requests view
  a. requests selector

Prio 2
- [ ] Reduce encoder model to bare essentials; expose minimal fields outside repository/crate.
- [ ] Make Indexer::dataset async (return BoxFuture) to support async dataset construction.

Prio 3
- [ ] Support stop/pause/resume by rethinking Requests -> Optimizations with OptimizationState + OptimizationBudget tables (event-sourced state, budget deltas).

Prio 4 — F1: Evaluation Noise Estimation
- [ ] Probe genotype generation and repeated evaluation via genotype_evaluation_service
- [ ] Rolling std dev convergence detection (epsilon + min_evaluations_per_probe)
- [ ] Budget exhaustion hard stop
- [ ] Per fitness-band noise floor reporting
- [ ] Stabilization chart output

Prio 5 — F2: Embedding Space Quality Diagnostics
- [ ] F2a: Fitness-Distance Correlation — rolling window, pairwise distances, Spearman correlation
- [ ] F2b: Neighborhood Fitness Variance — per-genotype multi-k computation

Prio 6 — Q1: Similarity-Based Evaluation Skipping
- [ ] Nearest-neighbor distance check before evaluation
- [ ] User-configurable threshold D and audit sample rate
- [ ] Skip error measurement in noise-floor units
- [ ] Convergence impact tracking

Prio 7 — Q3: Diversity-Aware Ensemble Identification
- [ ] Pareto-optimal set identification across fitness and behavioral dissimilarity

Prio 8 — Q2: Surrogate Modeling
- [ ] k-NN regressor over embedding for fitness rank prediction
- [ ] Dimensionality sweep to inform encoder selection

---

## Future Plans

### Q3 — Diversity-Aware Ensemble Identification *(blocked on F2 maturity)*
Identify a Pareto-optimal set of genotypes across fitness and pairwise
behavioral dissimilarity. Tooling lets the user assess whether the set forms a
useful ensemble in their domain. Feasibility depends on embedding space quality
surfaced by F2.

### Q2 — Surrogate Modeling *(blocked on Q1 and Q3)*
Use a k-NN regressor over the embedding to predict fitness rank and filter weak
candidates before evaluation. Feasibility depends on what Q1 and F2 reveal
about embedding structure. A dimensionality sweep (e.g. 32→256) would inform
encoder selection.
