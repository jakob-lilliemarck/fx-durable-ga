CREATE TABLE fx_durable_ga.individuals (
    genotype_id UUID PRIMARY KEY,
    request_id UUID NOT NULL,
    generation_id INTEGER NOT NULL
);

CREATE TABLE fx_durable_ga.fitness (
    genotype_id UUID PRIMARY KEY,
    fitness DOUBLE PRECISION NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL
);

CREATE VIEW fx_durable_ga.individuals_with_fitness AS
SELECT
    i.genotype_id,
    i.request_id,
    i.generation_id,
    f.fitness,
    f.evaluated_at
FROM fx_durable_ga.individuals i
LEFT JOIN fx_durable_ga.fitness f ON i.genotype_id = f.genotype_id;

CREATE VIEW fx_durable_ga.populations AS
SELECT
    request_id,
    COUNT(*) as total_individuals,
    COUNT(fitness) as evaluated_individuals,
    COUNT(*) - COUNT(fitness) as live_individuals,
    MAX(generation_id) as current_generation,
    MAX(fitness) as best_fitness
FROM fx_durable_ga.individuals_with_fitness
GROUP BY request_id;
