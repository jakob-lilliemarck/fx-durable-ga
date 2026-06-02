ALTER TABLE fx_durable_ga.requests ADD COLUMN type_hash INTEGER NOT NULL DEFAULT 0;
ALTER TABLE fx_durable_ga.genotypes ADD COLUMN type_hash INTEGER NOT NULL DEFAULT 0;
