# F1 — Evaluation Noise Diagnostic Service

## Overview

Estimate the noise floor of the evaluation function — variance in fitness when
evaluating the same genotype repeatedly. Expressed as mean, variance, and
standard deviation of fitness across multiple evaluations of one genotype.

## Service Structure

```
src/services/noise_diagnostics/
  mod.rs              -- Module declarations + pub use re-exports
  errors.rs           -- Error enum
  service.rs          -- Main service struct + impl
  registrations.rs    -- DI registration + job handler registration
  jobs.rs             -- Job messages + handlers
```

No `events.rs` — no event handlers needed. The diagnostic run is self-contained:
call `new_noise_probe`, jobs evaluate in background, results queried on-demand.

## Dependencies

| Dependency | Why |
|---|---|
| `Arc<evaluation::Service>` | Evaluate probe genotypes via `evaluate_genotype` |
| `genotypes::Read` | Fetch genotype by ID in job handler |
| `noise_diagnostics::Write` | Store probe records |
| `Arc<Queries>` | Dispatch MQ jobs |

## Schema

Replace the existing `20260320183321_noise_diagnostics` migration with:

```sql
CREATE TABLE fx_durable_ga.noise_probes (
    id UUID PRIMARY KEY,
    genotype_id UUID NOT NULL,
    request_id UUID NOT NULL,
    evaluation_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
```

A single table. Each row represents one probe — a request to evaluate a specific
genotype a set number of times. `request_id` is a grouping UUID provided by the
caller (e.g., a logical diagnostic run). Stats are queried on-demand from
`evaluation.evaluations` using `genotype_id` or `request_id`.

## Changes to Existing Code

### 1. Add `evaluation_id` to `GenotypeEvaluatedEvent`

**File:** `src/services/evaluation/events.rs`

```rust
pub struct GenotypeEvaluatedEvent {
    pub evaluation_id: Uuid,    // NEW
    pub request_id: Option<Uuid>,
    pub genotype_id: Uuid,
    pub fitness: f64,
}
```

**File:** `src/services/evaluation/service.rs`

Extract `evaluation.id()` before storing and pass it to the event:

```rust
let evaluation = Evaluation::new(/* ... */);
let evaluation_id = evaluation.id;
// ... store ...
publisher.publish(GenotypeEvaluatedEvent::new(
    evaluation_id,    // NEW
    genotype.request_id,
    genotype.id,
    fitness,
))
```

### 2. Add `get_evaluation_aggregates` query

**File:** `src/services/evaluation/repositories/evaluations/queries/get_evaluation_aggregates.rs`

New query file. Returns aggregate stats for evaluations matching optional
filters by `request_id` and `genotype_id`:

```rust
pub struct GetEvaluationAggregatesFilter {
    request_id: Option<Uuid>,
    genotype_id: Option<Uuid>,
}

impl GetEvaluationAggregatesFilter {
    pub fn with_request_id(mut self, id: Uuid) -> Self;
    pub fn with_genotype_id(mut self, id: Uuid) -> Self;
}

pub struct EvaluationAggregates {
    pub count: i64,
    pub avg_fitness: Option<f64>,
    pub stddev_fitness: Option<f64>,
    pub variance_fitness: Option<f64>,
}
```

SQL:
```sql
SELECT
    COUNT(*)::bigint AS "count!",
    AVG(fitness) AS "avg_fitness",
    STDDEV_POP(fitness) AS "stddev_fitness",
    VAR_POP(fitness) AS "variance_fitness"
FROM evaluation.evaluations
WHERE ($1::uuid IS NULL OR genotype_id = $1)
  AND ($2::uuid IS NULL OR request_id = $2)
```

**File:** `src/services/evaluation/repositories/evaluations/repository.rs`

Add to `impl Read`:
```rust
pub async fn get_evaluation_aggregates(
    &self,
    filter: &GetEvaluationAggregatesFilter,
) -> Result<EvaluationAggregates, Error>
```

Re-export `GetEvaluationAggregatesFilter` and `EvaluationAggregates` through
the evaluations module.

### 3. Noise diagnostics repository

Update `src/repositories/noise_diagnostics/` to work with the new single-table
schema. Replace the three old tables with just `noise_probes`:

- Models: `NoiseProbe { id, genotype_id, request_id, evaluation_count, created_at }`
- Read: `search_probes` with a `SearchNoiseProbesFilter` struct using the builder pattern
- Write: `store_probe`
- Queries: `store_noise_probe`, `get_noise_probe`, `get_noise_probes_by_request`

Follow the standard three-struct pattern: `WriteTx` with a `store_probe` method,
called inside `db::begin` alongside job dispatch.

## Service API

### `pub async fn new_noise_probe(
    &self,
    genotype_id: Uuid,
    request_id: Uuid,
    evaluation_count: i64,
) -> Result<Uuid, Error>`

The single entry point. Caller provides a probe genotype (already stored in the
genotypes table) and a grouping `request_id`.

```
new_noise_probe(genotype_id, request_id, evaluation_count):
    let probe_id = Uuid::now_v7()
    let jobs: Vec<EvaluateNoiseProbeGenotypeMessage> = (0..evaluation_count)
        .map(|_| EvaluateNoiseProbeGenotypeMessage::new(probe_id, genotype_id))
        .collect()

    db::begin(self.writer, |tx|:
        1. WriteTx.store_probe(NoiseProbe { id, genotype_id, request_id, evaluation_count })
        2. Publisher::new_tx(tx, mq).publish_many(&jobs)
    )

    return probe_id
```

Returns the `probe_id` so the caller can track the probe.

## Job Messages

### EvaluateNoiseProbeGenotypeMessage

```rust
pub(super) struct EvaluateNoiseProbeGenotypeMessage {
    pub probe_id: Uuid,
    pub genotype_id: Uuid,
}
```

### Handler Logic

```
handle(EvaluateNoiseProbeGenotypeMessage { probe_id, genotype_id }):
    1. Fetch genotype via genotypes_ro.get_genotype(genotype_id)
    2. evaluation_service.evaluate_genotype(
           genotype,
           drop_on=[],
           retry_on=[SHUTDOWN_SEMAPHORE],
       )
    3. Return Ok
```

If the application shuts down mid-evaluation, the job retries and produces
another data point. That is fine — more evaluations mean better noise estimates.

## Stats Retrieval (on-demand)

No persisted aggregate — computed from evaluations on-demand:

```rust
// Per-probe
let aggregates = evaluations_ro.get_evaluation_aggregates(
    &GetEvaluationAggregatesFilter::default()
        .with_genotype_id(probe.genotype_id)
);

// aggregates.count, aggregates.avg_fitness,
// aggregates.stddev_fitness, aggregates.variance_fitness

// Across all probes in a group
let aggregates = evaluations_ro.get_evaluation_aggregates(
    &GetEvaluationAggregatesFilter::default()
        .with_request_id(request_id)
);
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("genotype not found: {0}")]
    GenotypeNotFound(Uuid),

    #[error("evaluation_count must be positive, was {0}")]
    InvalidEvaluationCount(i64),

    #[error(transparent)]
    Database(#[from] sqlx::Error),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
```

## Test Plan

### Repository tests (`noise_diagnostics/repository.rs`)

- `store_probe` / `get_probe_by_id` round-trip
- `get_probes_by_request` returns all probes for a group
- `get_probe_by_id` returns None for unknown probe

### Evaluation aggregates query test (`queries.rs`)

- Returns aggregates for a genotype_id
- Returns zero count when no evaluations match
- Handles both request_id and genotype_id filters

### Service tests (`service.rs`)

- `new_noise_probe` creates probe + dispatches correct number of jobs

### Job handler tests (`jobs.rs`)

- Handler calls `evaluate_genotype` with correct genotype
- Handler passes shutdown semaphore in retry_on

## Migration

Edit `20260320183321_noise_diagnostics.up.sql` and `down.sql` in-place to
replace the three old tables (`noise_diagnostic_configs`,
`noise_diagnostic_runs`, `noise_diagnostic_run_evaluations`) and evaluation
schema changes with just the `noise_probes` table.
