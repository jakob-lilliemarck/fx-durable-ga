# Missing Service Tests

This document tracks service methods that lack test coverage.
Referenced from the AGENTS.md + SKILL.md review.

## Optimization Service (`src/services/optimization/service.rs`)

| Method | Visibility | Why it needs a test |
|---|---|---|
| `evaluate_genotype` | `pub(super)` | Handler entry point that orchestrates evaluation |
| `should_breed_next_generation` | `pub(super)` | Business decision logic for breeding |
| `try_charge` | `pub(super)` | Budget charge orchestration |
| `breed_genotypes` | `pub(super)` | Genotype breeding / crossover orchestration |
| `get_best_genotype` | `pub` | Public query method with business logic |
| `stop` | `pub` | Service shutdown logic |

## Genotype Indexing Service (`src/services/genotype_indexing/service.rs`)

| Method | Visibility | Why it needs a test |
|---|---|---|
| `retry_deferred_indexation` | `pub(super)` | Business method that retries deferred indexation |
