-- Drop GA-specific request config columns and add user_defined JSONB
ALTER TABLE fx_durable_ga.requests
  DROP COLUMN IF EXISTS crossover,
  DROP COLUMN IF EXISTS mutagen,
  DROP COLUMN IF EXISTS distribution;

ALTER TABLE fx_durable_ga.requests
  ADD COLUMN user_defined JSONB NOT NULL;