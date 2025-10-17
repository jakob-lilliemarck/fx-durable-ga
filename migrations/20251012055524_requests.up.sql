CREATE SCHEMA fx_durable_ga;

CREATE TABLE fx_durable_ga.requests (
    id UUID PRIMARY KEY,
    requested_at TIMESTAMPTZ NOT NULL,
    type_name TEXT NOT NULL,
    type_hash INTEGER NOT NULL,
    goal JSONB NOT NULL,
    schedule JSONB NOT NULL,
    selector JSONB NOT NULL,
    mutagen JSONB NOT NULL,
    crossover JSONB NOT NULL
);
