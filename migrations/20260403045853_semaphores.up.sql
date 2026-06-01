CREATE SCHEMA IF NOT EXISTS synchronization;

CREATE TABLE synchronization.semaphores (
    name TEXT NOT NULL,
    raised_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (name, raised_at)
);
