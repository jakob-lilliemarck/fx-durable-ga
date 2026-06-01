CREATE TABLE fx_durable_ga.noise_diagnostic_configs (
    id UUID PRIMARY KEY,
    optimization_type_name TEXT NOT NULL,
    probe_population_size INTEGER NOT NULL,
    probe_min_evaluations INTEGER NOT NULL,
    probe_max_evaluations INTEGER NOT NULL,
    budget_id UUID NOT NULL,
    revised_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE fx_durable_ga.noise_diagnostic_runs (
    id UUID PRIMARY KEY,
    noise_diagnostic_config_id UUID NOT NULL REFERENCES fx_durable_ga.noise_diagnostic_configs(id),
    initiated_at TIMESTAMPTZ NOT NULL
);

-- Step 1: Drop the copied_from column (removes the FK dependency on fitness_pkey)
ALTER TABLE fx_durable_ga.evaluations
    DROP COLUMN copied_from;

-- Step 2: Add id column
ALTER TABLE fx_durable_ga.evaluations
    ADD COLUMN id UUID;

-- Step 3: Populate
UPDATE fx_durable_ga.evaluations
    SET id = gen_random_uuid();

-- Step 4: Set NOT NULL
ALTER TABLE fx_durable_ga.evaluations
    ALTER COLUMN id SET NOT NULL;

-- Step 5: Drop the old PK
ALTER TABLE fx_durable_ga.evaluations
    DROP CONSTRAINT fitness_pkey;

-- Step 6: Add new PK
ALTER TABLE fx_durable_ga.evaluations
    ADD CONSTRAINT evaluations_pkey PRIMARY KEY (id);

ALTER TABLE fx_durable_ga.genotypes
    ALTER COLUMN request_id DROP NOT NULL,
    ALTER COLUMN generation_id DROP NOT NULL;

CREATE TABLE fx_durable_ga.noise_diagnostic_run_evaluations (
    noise_diagnostic_run_id UUID NOT NULL REFERENCES fx_durable_ga.noise_diagnostic_runs(id) ON DELETE CASCADE,
    evaluation_id UUID NOT NULL,
    referenced_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (noise_diagnostic_run_id, evaluation_id)
);
