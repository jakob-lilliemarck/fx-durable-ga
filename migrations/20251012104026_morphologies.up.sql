CREATE TABLE fx_durable_ga.morphologies (
    id UUID PRIMARY KEY,
    revised_at TIMESTAMPTZ NOT NULL,
    type_name TEXT NOT NULL,
    type_hash INTEGER NOT NULL
);

CREATE TABLE fx_durable_ga.gene_bounds (
    morphology_id UUID NOT NULL,
    position INTEGER NOT NULL,
    lower INTEGER NOT NULL,
    upper INTEGER NOT NULL,
    divisor INTEGER NOT NULL,
    PRIMARY KEY (morphology_id, position)
);
