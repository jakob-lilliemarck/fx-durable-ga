CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE fx_durable_ga.embeddings (
    id UUID PRIMARY KEY,
    encoder_digest CHAR(64) NOT NULL,
    encoded_at TIMESTAMPTZ NOT NULL,
    value VECTOR(256) NOT NULL
);

CREATE TABLE fx_durable_ga.embedding_tags (
    id UUID PRIMARY KEY,
    tag_hash BIGINT NOT NULL,
    tag_name TEXT NOT NULL,
    embedding_id UUID NOT NULL REFERENCES fx_durable_ga.embeddings(id) ON DELETE CASCADE,
    tagged_at TIMESTAMPTZ NOT NULL,
    UNIQUE(tag_hash, embedding_id)
);
CREATE INDEX idx_embedding_tags_hash ON fx_durable_ga.embedding_tags(tag_hash);
CREATE INDEX idx_embedding_tags_embedding ON fx_durable_ga.embedding_tags(embedding_id);

CREATE VIEW fx_durable_ga.tagged_embeddings AS
SELECT
    e.id,
    e.encoder_digest,
    e.encoded_at,
    e.value,
    t.tag_hash,
    t.tag_name,
    t.tagged_at
FROM fx_durable_ga.embeddings e
JOIN fx_durable_ga.embedding_tags t ON e.id = t.embedding_id;

CREATE INDEX idx_embeddings_ivfflat ON fx_durable_ga.embeddings
USING ivfflat (value vector_cosine_ops)
WITH (lists = 100);

CREATE TABLE fx_durable_ga.requested_embeddings (
    entity_id UUID NOT NULL,
    entity_type_name TEXT NOT NULL,
    indexer_digest CHAR(64) NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL,
    handled_at TIMESTAMPTZ,
    PRIMARY KEY (entity_id, indexer_digest)
);

CREATE INDEX idx_requested_embeddings_by_requested_at
ON fx_durable_ga.requested_embeddings (requested_at, entity_id)
WHERE handled_at IS NULL;

CREATE INDEX idx_requested_embeddings_by_indexer_digest
ON fx_durable_ga.requested_embeddings (indexer_digest, requested_at, entity_id)
WHERE handled_at IS NULL;
