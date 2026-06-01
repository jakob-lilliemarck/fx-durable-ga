CREATE SCHEMA IF NOT EXISTS budgeting;

CREATE TABLE budgeting.transactions (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL,
    account_type TEXT NOT NULL,
    amount BIGINT NOT NULL,
    balance BIGINT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    reason TEXT NOT NULL,
    CONSTRAINT delta_not_zero CHECK (amount != 0)
);

CREATE INDEX idx_transactions_budget_id ON budgeting.transactions (account_id, timestamp DESC);

ALTER TABLE fx_durable_ga.requests
ADD COLUMN account_id UUID;

UPDATE fx_durable_ga.requests
SET account_id = gen_random_uuid()
WHERE account_id IS NULL;

ALTER TABLE fx_durable_ga.requests
ALTER COLUMN account_id SET NOT NULL;

CREATE INDEX idx_requests_account_id ON fx_durable_ga.requests (account_id);
