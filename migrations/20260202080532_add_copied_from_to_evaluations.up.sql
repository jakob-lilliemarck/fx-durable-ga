ALTER TABLE fx_durable_ga.evaluations
    ADD COLUMN copied_from UUID REFERENCES fx_durable_ga.evaluations (genotype_id);
