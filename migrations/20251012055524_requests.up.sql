CREATE SCHEMA fx_durable_ga;

CREATE TYPE fx_durable_ga.fitness_goal AS ENUM ('minimize', 'maximize', 'exact');

CREATE TABLE fx_durable_ga.requests (
    id UUID PRIMARY KEY,
    requested_at TIMESTAMPTZ NOT NULL,
    type_name TEXT NOT NULL,
    type_hash INTEGER NOT NULL,
    goal fx_durable_ga.fitness_goal NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    strategy JSONB NOT NULL,
    temperature DOUBLE PRECISION NOT NULL,
    mutation_rate DOUBLE PRECISION NOT NULL
);
