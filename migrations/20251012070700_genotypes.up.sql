CREATE TABLE fx_durable_ga.genotypes (
    id UUID PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL,
    type_name TEXT NOT NULL,
    type_hash INTEGER NOT NULL,
    genome BIGINT[] NOT NULL,
    request_id UUID NOT NULL,
    fitness DOUBLE PRECISION,
    generation_id INTEGER NOT NULL
);

CREATE TABLE fx_durable_ga.populations (
    request_id UUID NOT NULL,
    genotype_id UUID NOT NULL,
    PRIMARY KEY (request_id, genotype_id)
);
