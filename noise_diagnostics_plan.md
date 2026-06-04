# Noise Diagnostics — Completion Plan

## Current state

The `noise_diagnostics` service exists with:

- `Service::new_noise_probe(genotype_id, request_id, evaluation_count)` —
  stores a `NoiseProbe` record and dispatches N evaluation jobs atomically
- `Service::evaluate_probe(probe_id, genotype)` — evaluates the probe genotype
  (called by the job handler)
- `NoiseProbe` model: `id`, `genotype_id`, `request_id`, `evaluation_count`,
  `created_at`
- `Read::search_noise_probes(filter)` — filter by `request_id` or `genotype_id`
- Full job handler (`EvaluateNoiseProbeGenotype`) and test suite

What's missing: a way to **query results** — completion progress and noise
floor estimates.

## Scope

No new tables, no convergence detection, no budget integration. The
`evaluations` table is the sole source of truth for noise estimates.

## Work items

### 1. `probe_noise_floor(probe_id)` → `NoiseFloor`

A service method that:
1. Looks up the probe via `probes::Read::search_noise_probes`
2. Queries `evaluations::Read::get_evaluation_aggregates` filtered by
   `group_id = probe_id`
3. Returns a struct with:
   - `sample_count` / `target_count` (completion progress)
   - `mean_fitness` / `stddev_fitness` (noise floor estimate)
   - `converged: bool` — true when `sample_count >= target_count`

```
NoiseFloor {
    probe_id: Uuid,
    genotype_id: Uuid,
    sample_count: i64,
    target_count: i32,
    mean_fitness: Option<f64>,
    stddev_fitness: Option<f64>,
    converged: bool,
}
```

### 2. `probe_noise_floors(request_id)` → `Vec<NoiseFloor>`

Batch version — queries all probes for a given `request_id`, then queries
evaluation aggregates for each probe. Simple sequential composition in the
service layer.

### 3. `fitness_band_noise_floors(request_id, band_count)` → `Vec<BandNoiseFloor>`

Groups probes by fitness range and aggregates noise floor per band:

1. Fetch all probes for `request_id`
2. For each probe, resolve the genotype to get its fitness (last evaluation)
3. Divide the fitness range into `band_count` equal bands
4. For each band, average the noise floors of probes in that band
5. Return `BandNoiseFloor { band_index, fitness_range, probe_count, mean_noise, stddev_noise }`

This is a pure service-layer computation — no new queries needed.

## Dependencies

- `evaluations::Read` — already injectable; the service already uses
  `genotypes::Read` and `evaluation::Service`
- `probes::Read` — already injectable; currently only `Write` is wired (for
  `new_noise_probe`); need to add `Read` to the service constructor

## Files to change

| File | Change |
|------|--------|
| `src/services/noise_diagnostics/service.rs` | Add `probes_ro: probes::Read` field; add `probe_noise_floor`, `probe_noise_floors`, `fitness_band_noise_floors` methods |
| `src/services/noise_diagnostics/registrations.rs` | Inject `probes::Read` into the service |
| `src/services/noise_diagnostics/mod.rs` | Re-export `NoiseFloor` / `BandNoiseFloor` if needed |
| `src/services/noise_diagnostics/service.rs` (tests) | Add tests for all three methods |

## Non-goals (explicitly excluded)

- No `noise_estimates` table
- No convergence detection / rolling std dev
- No budget exhaustion hard stop
- No stabilization chart data (individual fitness samples are in evaluations)
