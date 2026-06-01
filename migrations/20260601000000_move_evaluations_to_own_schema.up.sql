CREATE SCHEMA IF NOT EXISTS evaluation;

CREATE TABLE evaluation.evaluations (
    id UUID PRIMARY KEY,
    genotype_id UUID NOT NULL REFERENCES fx_durable_ga.genotypes(id),
    fitness DOUBLE PRECISION NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    evaluated_by UUID,
    request_id UUID,
    generated_at TIMESTAMPTZ
);

INSERT INTO evaluation.evaluations
SELECT
    e.id, e.genotype_id, e.fitness, e.started_at, e.completed_at, e.evaluated_by,
    g.request_id, g.generated_at
FROM fx_durable_ga.evaluations e
LEFT JOIN fx_durable_ga.genotypes g ON e.genotype_id = g.id;

DROP VIEW IF EXISTS fx_durable_ga.populations;

DROP TABLE fx_durable_ga.evaluations;

CREATE INDEX idx_evaluations_genotype_id ON evaluation.evaluations (genotype_id);
CREATE INDEX idx_evaluations_request_id ON evaluation.evaluations (request_id);
CREATE INDEX idx_evaluations_request_id_gen ON evaluation.evaluations (request_id, generated_at, id);
CREATE INDEX idx_evaluations_fitness ON evaluation.evaluations (fitness);

ALTER TABLE fx_durable_ga.requests DROP COLUMN user_defined, DROP COLUMN data;
