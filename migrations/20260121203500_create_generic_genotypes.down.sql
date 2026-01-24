ALTER TABLE fx_durable_ga.genotypes
    ALTER COLUMN genome TYPE BIGINT[]
    USING ARRAY[]::bigint[];
