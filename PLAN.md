# Problem statement
Duplicated genomes currently trigger parent re-selection in `Service::breed_genotypes`, introducing selection bias and causing extra evaluations. We need to redesign deduplication so that the same parents simply retry crossover/mutation, and only copy evaluation results when uniqueness fails. Supporting repo changes must let us batch-write evaluations, fetch original evaluations deterministically, and capture copied-evaluation metadata. Timings statistics must also ignore copied duplicates.
# Current state
`record_evaluations` is unimplemented, so evaluations can only be recorded individually. `search` (in `src/repositories/genotypes/queries.rs`) returns `(Genotype, Option<f64>)` but lacks filtering by genome hash and does not guarantee access to evaluation metadata. `get_intersection` retrieves duplicate genomes but lacks ordering to find the original evaluation. The evaluations table lacks a way to mark copied rows. `breed_genotypes` currently re-selects parents when collisions occur, leading to biased sampling. Timings aggregation (`get_timings`) counts every evaluation row, so cloned entries will skew duration stats. `Service::max_deduplication_attempts` currently limits re-selection loops.
# Proposed changes
## Repository layer
1. Implement `record_evaluations(tx, evaluations: Vec<Evaluation>) -> Result<Vec<Evaluation>, Error>` using `sqlx::QueryBuilder` to batch insert and return the inserted rows. Update tests to cover success and conflict scenarios.
2. Leave `search` untouched so its behavior stays backward compatible.
3. Extend `get_intersection` so it uses `DISTINCT ON (genome_hash) ... ORDER BY completed_at ASC` and LEFT JOINs evaluations, ensuring we always retrieve the earliest/original evaluation per hash. Return `Vec<(Genotype, Option<Evaluation>)>` so the service can grab fitness and metadata in one call. Add/adjust tests for the new behavior.
4. Add a nullable `copied_from` column to `fx_durable_ga.evaluations` via a new migration (plus model and repository updates). When inserting a copied evaluation we set `copied_from = Some(original_evaluation_id)`; original evaluations keep it `NULL`.
5. Update `get_timings` to aggregate only distinct genomes: use `DISTINCT ON (g.genome_hash)` ordered by `e.completed_at ASC` so copied evaluations do not skew timing stats. Adjust tests covering timing summaries.
## Service changes (`src/services/optimization/service.rs`)
1. Refactor `breed_genotypes` so the main loop never re-selects parents. For each parent pair, run crossover+mutation up to `self.max_deduplication_attempts` times:
    * Each attempt checks previously generated hashes and existing DB hashes (via the updated intersection query) before accepting the child.
    * On every duplicate attempt, log a warning (`tracing::warn!`), including parent IDs and attempt count for observability.
2. If all attempts fail for a given child slot:
    * Log another warning indicating retries were exhausted and we will copy an evaluation.
    * Do not enqueue jobs for those genome hashes.
    * Use the enhanced `get_intersection` result to find the original evaluation (the earliest one per hash). Create a new `Evaluation` with the duplicate genotype’s ID, copy the fitness, set `started_at`/`completed_at` to `Utc::now()`, `evaluated_by` to `self.host_id`, and `copied_from` to the original evaluation’s ID.
    * Batch-write these copied evaluations using the new `record_evaluations` helper so subsequent queries see the fitness immediately.
3. Ensure we still publish `GenotypeGenerated` events only for genuinely inserted genotypes. For those that receive copied evaluations, skip dispatching evaluation jobs but persist the evaluation row so the scheduler views them as completed.
## Logging and configuration
1. Reinterpret `Service::max_deduplication_attempts` as the crossover/mutation retry cap (no parent reselection). Update any documentation/comments accordingly.
2. Add `tracing::warn!` logs for both duplicate-detected retries and exhausted-attempts copy operations, so operators can monitor clone pressure without a dedicated table yet.
## Testing
1. Add repository-level tests for `record_evaluations` batch insertion and conflicts.
2. Extend intersection/search tests to cover `with_genome_hashes` and `DISTINCT ON` ordering with evaluations.
3. Add service-level or integration tests (if feasible) covering the new retry-and-copy flow, ensuring that duplicates reuse fitness without spawning jobs.
4. Update timing summary tests to reflect the new DISTINCT ON logic and ensure counts exclude copied evaluations.
## Migration plan
1. Create a new SQL migration adding `copied_from uuid NULL REFERENCES fx_durable_ga.evaluations(genotype_id)` (or a dedicated PK if available) and any necessary indexes.
2. Regenerate query bindings/types if required by `sqlx` macros.
3. Document the new semantics (copied evaluations, logging) in code comments where appropriate.
# Risks and follow-ups
* Need to ensure `copied_from` references the correct evaluation identifier (if `evaluations` lack a dedicated ID, we may need to reference `(genotype_id, completed_at)` or introduce a surrogate key).
* Service retries must guard against infinite loops when mutation deterministically outputs the same genome; the capped attempts mitigate this but require careful handling of RNG seeding.
* Metrics consumers should be notified that timing data now reflects earliest evaluations only; future work could still add the `duplicate_genomes` table for richer telemetry.
