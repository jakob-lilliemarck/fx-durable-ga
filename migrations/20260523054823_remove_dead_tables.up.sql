DROP VIEW fx_durable_ga.enabled_encoder_digests;

DROP TABLE fx_durable_ga.request_conclusions;
DROP TABLE fx_durable_ga.morphologies;
DROP TABLE fx_durable_ga.encoder_toggles;

-- adds a view to use in place of dropped fx_durable_ga.enabled_encoder_digests
CREATE VIEW fx_durable_ga.available_encoder_digests AS
SELECT digest FROM (
    SELECT DISTINCT ON (digest) ea.digest, ea.is_available
    FROM fx_durable_ga.encoder_availability ea
    ORDER BY digest, revised_at DESC
) AS latest
WHERE is_available = TRUE;
