-- 1. Add nullable column, no default
ALTER TABLE fx_durable_ga.requested_embeddings
    ADD COLUMN IF NOT EXISTS metadata JSONB;

-- 2. Backfill existing rows
UPDATE fx_durable_ga.requested_embeddings SET metadata = '{}' WHERE metadata IS NULL;

-- 3. Now safe to add NOT NULL
ALTER TABLE fx_durable_ga.requested_embeddings ALTER COLUMN metadata SET NOT NULL;
