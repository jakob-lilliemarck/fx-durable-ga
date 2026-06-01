DROP VIEW IF EXISTS fx_durable_ga.populations;

CREATE TABLE fx_durable_ga.evaluations (
    id UUID PRIMARY KEY,
    genotype_id UUID NOT NULL REFERENCES fx_durable_ga.genotypes(id),
    fitness DOUBLE PRECISION NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    evaluated_by UUID
);

INSERT INTO fx_durable_ga.evaluations (id, genotype_id, fitness, started_at, completed_at, evaluated_by)
SELECT id, genotype_id, fitness, started_at, completed_at, evaluated_by
FROM evaluation.evaluations;

DROP SCHEMA IF EXISTS evaluation CASCADE;

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
