CREATE TABLE fx_durable_ga.morphologies (
    type_name TEXT NOT NULL,
    type_hash INTEGER NOT NULL,
    revised_at TIMESTAMPTZ NOT NULL,
    gene_bounds JSONB NOT NULL,
    PRIMARY KEY (type_hash, revised_at)
);
