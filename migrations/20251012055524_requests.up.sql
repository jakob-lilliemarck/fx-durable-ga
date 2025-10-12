CREATE SCHEMA fx_durable_ga;

CREATE TYPE fx_durable_ga.fitness_goal AS ENUM ('minimize', 'maximize', 'exact');

CREATE TABLE fx_durable_ga.requests (
    id UUID PRIMARY KEY,
    requested_at TIMESTAMPTZ NOT NULL,
    name TEXT NOT NULL,
    hash INTEGER NOT NULL,
    goal fx_durable_ga.fitness_goal NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    max_generations BIGINT NOT NULL,
    population_size BIGINT NOT NULL
);
