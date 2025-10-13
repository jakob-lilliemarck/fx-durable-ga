CREATE TABLE fx_durable_ga.populations (
    request_id UUID NOT NULL,
    genotype_id UUID NOT NULL,
    PRIMARY KEY (request_id, genotype_id)
);
