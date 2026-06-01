DROP INDEX IF EXISTS request_conclusions_id_idx;

ALTER TABLE fx_durable_ga.request_conclusions
DROP COLUMN IF EXISTS id;
