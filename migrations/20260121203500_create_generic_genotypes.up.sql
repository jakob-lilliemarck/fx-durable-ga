-- Alter the canonical genotypes table to use JSONB genomes.
ALTER TABLE fx_durable_ga.genotypes
    ALTER COLUMN genome TYPE JSONB
    USING to_jsonb(genome);
