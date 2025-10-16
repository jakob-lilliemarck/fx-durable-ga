CREATE TABLE fx_durable_ga.genotypes (
    id UUID PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL,
    type_name TEXT NOT NULL,
    type_hash INTEGER NOT NULL,
    genome BIGINT[] NOT NULL,
    request_id UUID NOT NULL REFERENCES fx_durable_ga.requests(id),
    generation_id INTEGER NOT NULL
);

CREATE TABLE fx_durable_ga.fitness (
    genotype_id UUID PRIMARY KEY REFERENCES fx_durable_ga.genotypes(id),
    fitness DOUBLE PRECISION NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL
);

CREATE VIEW fx_durable_ga.populations AS
SELECT
    request_id,
    COUNT(f.fitness) as evaluated_genotypes,
    COUNT(*) - COUNT(f.fitness) as live_genotypes,
    MAX(generation_id) as current_generation,
    MAX(f.fitness) as best_fitness
FROM fx_durable_ga.genotypes g
LEFT JOIN fx_durable_ga.fitness f ON g.id = f.genotype_id
GROUP BY request_id;
