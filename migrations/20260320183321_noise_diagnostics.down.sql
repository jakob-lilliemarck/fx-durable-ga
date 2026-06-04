-- Drop new tables
DROP TABLE IF EXISTS fx_durable_ga.noise_diagnostic_run_evaluations;
DROP TABLE IF EXISTS fx_durable_ga.noise_diagnostic_runs;
DROP TABLE IF EXISTS fx_durable_ga.noise_diagnostic_configs;
DROP TABLE IF EXISTS fx_durable_ga.noise_probes;

-- Restore NOT NULL on genotypes
ALTER TABLE fx_durable_ga.genotypes
    ALTER COLUMN generation_id SET NOT NULL;

-- Revert evaluations PK back to genotype_id
ALTER TABLE fx_durable_ga.evaluations
    DROP CONSTRAINT evaluations_pkey;

ALTER TABLE fx_durable_ga.evaluations
    ADD CONSTRAINT fitness_pkey PRIMARY KEY (genotype_id);

-- Drop the id column
ALTER TABLE fx_durable_ga.evaluations
    DROP COLUMN id;

-- Restore copied_from column
ALTER TABLE fx_durable_ga.evaluations
    ADD COLUMN copied_from UUID REFERENCES fx_durable_ga.evaluations(genotype_id);
