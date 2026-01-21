-- A new table for storing generic, serializable genotypes as JSONB.
-- This runs in parallel to the existing `genotypes` table to avoid disruption.
CREATE TABLE fx_durable_ga.generic_genotypes (
    id UUID PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL,
    type_name TEXT NOT NULL,
    type_hash INTEGER NOT NULL,
    genome_data JSONB,
    genome_hash BIGINT NOT NULL,
    request_id UUID NOT NULL REFERENCES fx_durable_ga.requests(id),
    generation_id INTEGER NOT NULL
);

-- Indexes to support efficient querying, mirroring the original genotypes table.
CREATE INDEX idx_generic_genotypes_request_generation ON fx_durable_ga.generic_genotypes (request_id, generation_id);
CREATE INDEX idx_generic_genotypes_request_genome_hash ON fx_durable_ga.generic_genotypes (request_id, genome_hash);
