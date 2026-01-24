ALTER TABLE fx_durable_ga.requests
  DROP COLUMN IF EXISTS user_defined;

ALTER TABLE fx_durable_ga.requests
  ADD COLUMN crossover JSONB,
  ADD COLUMN mutagen JSONB,
  ADD COLUMN distribution JSONB;