-- Add up migration script here
CREATE INDEX idx_genotypes_request_generated_at_id
    ON fx_durable_ga.genotypes (request_id, generated_at, id);
