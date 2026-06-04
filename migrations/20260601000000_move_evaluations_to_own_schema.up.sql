CREATE SCHEMA IF NOT EXISTS evaluation;

CREATE TABLE evaluation.evaluations (
    id UUID PRIMARY KEY,
    genotype_id UUID NOT NULL REFERENCES fx_durable_ga.genotypes(id),
    fitness DOUBLE PRECISION NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    evaluated_by UUID,
    group_id UUID NOT NULL,
    reason TEXT NOT NULL,
    generated_at TIMESTAMPTZ
);

DROP VIEW IF EXISTS fx_durable_ga.populations;

DROP TABLE fx_durable_ga.evaluations;

CREATE INDEX idx_evaluations_genotype_id ON evaluation.evaluations (genotype_id);
CREATE INDEX idx_evaluations_group_id ON evaluation.evaluations (group_id);
CREATE INDEX idx_evaluations_group_id_gen ON evaluation.evaluations (group_id, generated_at, id);
CREATE INDEX idx_evaluations_fitness ON evaluation.evaluations (fitness);

ALTER TABLE fx_durable_ga.requests DROP COLUMN user_defined, DROP COLUMN data;
