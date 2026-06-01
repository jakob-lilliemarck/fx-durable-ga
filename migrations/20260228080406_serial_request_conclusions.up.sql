ALTER TABLE fx_durable_ga.request_conclusions
ADD COLUMN serial BIGSERIAL;

WITH ordered AS (
    SELECT request_id
    FROM fx_durable_ga.request_conclusions
    ORDER BY concluded_at ASC, request_id ASC
),
numbered AS (
    SELECT request_id, row_number() OVER () AS seq
    FROM ordered
)
UPDATE fx_durable_ga.request_conclusions rc
SET serial = numbered.seq
FROM numbered
WHERE rc.request_id = numbered.request_id
  AND rc.serial IS NULL;

ALTER TABLE fx_durable_ga.request_conclusions
ALTER COLUMN serial SET NOT NULL;

CREATE UNIQUE INDEX request_conclusions_id_idx
ON fx_durable_ga.request_conclusions (serial);
