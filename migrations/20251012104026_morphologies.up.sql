CREATE TABLE fx_durable_ga.morphologies (
    revised_at TIMESTAMPTZ NOT NULL,
    type_name TEXT NOT NULL,
    type_hash INTEGER PRIMARY KEY
);

CREATE TABLE fx_durable_ga.gene_bounds (
    morphology_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    lower INTEGER NOT NULL,
    upper INTEGER NOT NULL,
    divisor INTEGER NOT NULL,
    PRIMARY KEY (morphology_id, position)
);
