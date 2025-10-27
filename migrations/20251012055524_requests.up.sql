CREATE SCHEMA IF NOT EXISTS fx_durable_ga;

CREATE TABLE fx_durable_ga.requests (
    id UUID PRIMARY KEY,
    requested_at TIMESTAMPTZ NOT NULL,
    type_name TEXT NOT NULL,
    type_hash INTEGER NOT NULL,
    goal JSONB NOT NULL,
    schedule JSONB NOT NULL,
    selector JSONB NOT NULL,
    mutagen JSONB NOT NULL,
    crossover JSONB NOT NULL,
    distribution JSONB NOT NULL
);

CREATE TYPE fx_durable_ga.conclusion AS ENUM (
    'completed',
    'terminated'
);

CREATE TABLE fx_durable_ga.request_conclusions (
    request_id UUID PRIMARY KEY REFERENCES fx_durable_ga.requests(id),
    concluded_at TIMESTAMPTZ NOT NULL,
    concluded_with fx_durable_ga.conclusion NOT NULL
);
