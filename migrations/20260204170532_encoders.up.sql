CREATE TABLE encoders (
    id UUID PRIMARY KEY,
    model_type TEXT NOT NULL,
    model_config JSONB NOT NULL,
    model_weights BYTEA NOT NULL,
    model_format TEXT NOT NULL,
    shape_in INTEGER[] NOT NULL,
    shape_out INTEGER NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL,
    trained_on_checksum BYTEA NOT NULL
);
