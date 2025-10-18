-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS fx_durable_ga.request_conclusions;
DROP TABLE IF EXISTS fx_durable_ga.requests;

-- Drop enum type
DROP TYPE IF EXISTS fx_durable_ga.conclusion;

-- Drop schema
DROP SCHEMA IF EXISTS fx_durable_ga;
