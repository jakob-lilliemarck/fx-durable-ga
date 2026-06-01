CREATE TABLE fx_durable_ga.encoders (
    digest CHAR(64) PRIMARY KEY,
    encodable_type_name TEXT NOT NULL,
    model_config JSONB NOT NULL,
    model_weights BYTEA NOT NULL,
    model_format TEXT NOT NULL,
    shape_in INTEGER[] NOT NULL,
    shape_out INTEGER NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL
);

-- Encoder toggle state
CREATE TABLE fx_durable_ga.encoder_toggles (
    digest CHAR(64) NOT NULL,
    is_enabled BOOLEAN NOT NULL,
    toggled_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (digest, toggled_at),
    FOREIGN KEY (digest) REFERENCES fx_durable_ga.encoders(digest)
);

CREATE INDEX idx_encoder_toggles_latest
ON fx_durable_ga.encoder_toggles (digest, toggled_at DESC);

CREATE VIEW fx_durable_ga.enabled_encoder_digests AS
SELECT digest FROM (
    SELECT DISTINCT ON (digest) et.digest, et.is_enabled
    FROM fx_durable_ga.encoder_toggles et
    ORDER BY digest, toggled_at DESC
) AS latest
WHERE is_enabled = TRUE;

-- Encoder availability
CREATE TABLE fx_durable_ga.encoder_availability (
    digest CHAR(64) NOT NULL,
    is_available BOOLEAN NOT NULL,
    revised_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (digest, revised_at)
);

CREATE INDEX idx_encoder_availability_latest
ON fx_durable_ga.encoder_availability (digest, revised_at DESC);

CREATE VIEW fx_durable_ga.latest_encoder_availability AS
SELECT DISTINCT ON (digest) ea.digest, ea.is_available
FROM fx_durable_ga.encoder_availability ea
ORDER BY digest, revised_at DESC;
