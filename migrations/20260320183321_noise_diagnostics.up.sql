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

CREATE TABLE IF NOT EXISTS fx_durable_ga.noise_probes (
    id UUID PRIMARY KEY,
    genotype_id UUID NOT NULL,
    request_id UUID NOT NULL,
    evaluation_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
