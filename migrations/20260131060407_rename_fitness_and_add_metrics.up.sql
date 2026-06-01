-- Drops dependent view so we can rename the table
DROP VIEW IF EXISTS fx_durable_ga.populations;

-- Renames fitness to evaluations
ALTER TABLE fx_durable_ga.fitness
    RENAME TO evaluations;

-- Adds timing columns
ALTER TABLE fx_durable_ga.evaluations
    DROP COLUMN evaluated_at,
    ADD COLUMN started_at TIMESTAMPTZ,
    ADD COLUMN completed_at TIMESTAMPTZ,
    ADD COLUMN evaluated_by UUID;

-- Adds parent columns (nullable, referencing genotypes)
ALTER TABLE fx_durable_ga.genotypes
    ADD COLUMN parent_a UUID REFERENCES fx_durable_ga.genotypes (id),
    ADD COLUMN parent_b UUID REFERENCES fx_durable_ga.genotypes (id);

-- Add indexes to aid recursive lookup of parents
CREATE INDEX genotypes_parent_a_idx ON fx_durable_ga.genotypes (parent_a);
CREATE INDEX genotypes_parent_b_idx ON fx_durable_ga.genotypes (parent_b);

-- Recreate view pointing to evaluations
CREATE VIEW fx_durable_ga.populations AS
SELECT
    request_id,
    COUNT(e.fitness) AS evaluated_genotypes,
    COUNT(*) - COUNT(e.fitness) AS live_genotypes,
    MAX(generation_id) AS current_generation,
    MIN(e.fitness) AS min_fitness,
    MAX(e.fitness) AS max_fitness
FROM fx_durable_ga.genotypes g
LEFT JOIN fx_durable_ga.evaluations e ON g.id = e.genotype_id
GROUP BY request_id;
