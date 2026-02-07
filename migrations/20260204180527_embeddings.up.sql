CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
    id UUID PRIMARY KEY,
    encoded_with UUID NOT NULL REFERENCES fx_durable_ga.encoders(id),
    encoded_at TIMESTAMPTZ NOT NULL,
    value VECTOR(256) NOT NULL
);

CREATE TABLE embedding_tags (
    tag_hash BIGINT NOT NULL,
    tag_name TEXT NOT NULL,
    embedding_id UUID NOT NULL REFERENCES fx_durable_ga.embeddings(id) ON DELETE CASCADE,
    tagged_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (embedding_id, tag_hash)
);

CREATE INDEX idx_embedding_tags_hash ON embedding_tags(tag_hash);
CREATE INDEX idx_embedding_tags_embedding ON embedding_tags(embedding_id);

CREATE VIEW tagged_embeddings AS
SELECT
    e.id,
    e.encoded_with,
    e.encoded_at,
    e.value,
    t.tag_hash,
    t.tag_name,
    t.tagged_at
FROM embeddings e
JOIN embedding_tags t ON e.id = t.embedding_id;

CREATE INDEX idx_embeddings_ivfflat ON embeddings
USING ivfflat (value vector_cosine_ops)
WITH (lists = 100);
