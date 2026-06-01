-- Drop the view that depends on evaluations
DROP VIEW IF EXISTS fx_durable_ga.populations;

-- Drop the indexes
DROP INDEX IF EXISTS genotypes_parent_b_idx;
DROP INDEX IF EXISTS genotypes_parent_a_idx;

-- Remove parent columns
ALTER TABLE fx_durable_ga.genotypes
    DROP COLUMN parent_b,
    DROP COLUMN parent_a;

-- Remove timing columns
ALTER TABLE fx_durable_ga.evaluations
    DROP COLUMN completed_at,
    DROP COLUMN started_at,
    ADD COLUMN evaluated_at TIMESTAMPTZ;

-- Rename table back to fitness
ALTER TABLE fx_durable_ga.evaluations
    RENAME TO fitness;

-- Recreate view referencing fitness
CREATE VIEW fx_durable_ga.populations AS
SELECT
    request_id,
    COUNT(f.fitness) AS evaluated_genotypes,
    COUNT(*) - COUNT(f.fitness) AS live_genotypes,
    MAX(generation_id) AS current_generation,
    MIN(f.fitness) AS min_fitness,
    MAX(f.fitness) AS max_fitness
FROM fx_durable_ga.genotypes g
LEFT JOIN fx_durable_ga.fitness f ON g.id = f.genotype_id
GROUP BY request_id;
