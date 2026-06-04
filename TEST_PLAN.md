# Test Plan — Noise Diagnostics

## Existing tests (6)

| Test | File | Layer |
|---|---|---|
| `store_and_search_by_request_id` | `queries.rs` | Repository query |
| `search_by_genotype_id` | `queries.rs` | Repository query |
| `search_returns_empty_for_unknown_request` | `queries.rs` | Repository query |
| `new_noise_probe_returns_error_for_invalid_count` | `service.rs` | Service |
| `new_noise_probe_dispatches_correct_number_of_jobs` | `service.rs` | Service |
| `noise_probe_evaluations_do_not_trigger_maintain_population` | `service.rs` | Service + event isolation |

## Proposed tests

### Test A: `search_by_request_id_returns_all_probes`

**File:** `src/services/noise_diagnostics/repositories/probes/queries.rs`
**Layer:** Repository query test (inline `#[cfg(test)]`)

**Rationale:** Tests the `request_id` filter on the search query returns the correct subset of probes. This is a query-level test — no DI container, constructs `Read` directly.

**Setup:**
- `store_noise_probe()` × 3 (two with shared `request_id`, one with different)

**Act:**
- `search_noise_probes(&pool, &SearchNoiseProbesFilter::default().with_request_id(shared_id))`

**Assert:**
- Returns 2 probes
- Both have `request_id == shared_id`

### Test B: `noise_stats_queryable_after_evaluation`

**File:** `src/services/noise_diagnostics/service.rs`
**Layer:** Service integration test (inline `#[cfg(test)]`)

**Rationale:** Tests the full pipeline: probe creation → evaluation → stats query. No event listener needed — stats are read directly from `evaluation.evaluations` via `get_evaluation_aggregates`. No arbitrary sleep needed.

**Setup:**
- Build container with `TestConfig::new(pool.clone()).build()` (no listeners)
- Register `TestEvaluator` on evaluation service
- Create request (FK) and probe genotype
- Call `new_noise_probe(genotype_id, request_id, 1)`
- Fetch the job from MQ, get genotype from payload
- Call `evaluate_probe(probe_id, genotype)`
- Get `evaluations::Read` from container

**Act:**
- `evaluations_ro.get_evaluation_aggregates(&GetEvaluationAggregatesFilter::default().with_genotype_id(genotype_id))`

**Assert:**
- `result.count == 1`
- `result.avg_fitness == Some(0.5)`
- `result.stddev_fitness == None`
- `result.variance_fitness == None`

The `TestEvaluator` struct and `create_request` helper are already defined in the test module — no new types needed.
