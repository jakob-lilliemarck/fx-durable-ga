use crate::repositories::embeddings::{
    Embedding,
    repository::{Similar, Tag},
};
use chrono::{DateTime, Utc};
use pgvector::Vector;
use sqlx::{PgExecutor, Row};
use uuid::Uuid;

pub struct DbEmbedding {
    pub(crate) id: Uuid,
    pub(crate) encoded_at: DateTime<Utc>,
    pub(crate) encoded_with: Uuid,
    pub(crate) value: pgvector::Vector,
}

pub(crate) async fn get_similar<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: &Uuid,
    tag_hashes: &[i64],
    limit: i64,
) -> Result<Vec<Similar>, super::Error> {
    let similar = sqlx::query_as!(
        Similar,
        r#"
        WITH query_embedding AS (
            SELECT value
            FROM embeddings
            WHERE id = $1
        )
        SELECT DISTINCT ON (te.id)
            te.id AS "id!: Uuid",
            te.value <=> qe.value AS "distance!: f64"
        FROM tagged_embeddings te
        CROSS JOIN query_embedding qe
        WHERE te.tag_hash = ANY ($2)
        ORDER BY te.id, "distance!: f64"
        LIMIT $3
        "#,
        id,
        tag_hashes,
        limit
    )
    .fetch_all(tx)
    .await?;

    Ok(similar)
}

#[cfg(test)]
mod tests_get_similar {
    #[sqlx::test(migrations = false)]
    async fn it_gets_similar_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        todo!()
    }
}

impl TryFrom<DbEmbedding> for Embedding {
    type Error = super::Error;

    fn try_from(value: DbEmbedding) -> Result<Self, Self::Error> {
        let embedding: [f32; 256] = value
            .value
            .to_vec()
            .try_into()
            .map_err(|v: Vec<f32>| super::Error::InvalidDimension(v.len()))?;

        Ok(Self {
            id: value.id,
            encoded_at: value.encoded_at,
            encoded_with: value.encoded_with,
            value: embedding,
        })
    }
}

pub(crate) async fn store_embedding<'tx, E: PgExecutor<'tx>>(
    tx: E,
    embedding: &Embedding,
) -> Result<Embedding, super::Error> {
    let value = Vector::from(embedding.value.to_vec());

    let embedding = sqlx::query_as!(
        DbEmbedding,
        r#"
            INSERT INTO fx_durable_ga.embeddings (
                id,
                encoded_at,
                encoded_with,
                value
            )
            VALUES ($1, $2, $3, $4)
            RETURNING
                id,
                encoded_at,
                encoded_with,
                value as "value: Vector"
            "#,
        embedding.id,
        embedding.encoded_at,
        embedding.encoded_with,
        value as Vector
    )
    .fetch_one(tx)
    .await?
    .try_into()?;

    Ok(embedding)
}

#[cfg(test)]
mod tests_store_embedding {
    #[sqlx::test(migrations = false)]
    async fn it_stores_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        todo!()
    }
}

pub type TagRow<'a> = (i64, &'a str, &'a Uuid, &'a DateTime<Utc>);

pub(crate) async fn store_tags<'tx, E, I>(tx: E, tags: I) -> Result<Vec<Tag>, super::Error>
where
    E: PgExecutor<'tx>,
    I: IntoIterator<Item = TagRow<'tx>>,
{
    let mut tags = tags.into_iter().peekable();
    if tags.peek().is_none() {
        return Ok(vec![]);
    }

    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.embedding_tags (
            tag_hash,
            tag_name,
            embedding_id,
            tagged_at
        )
        VALUES ",
    );

    let mut first = true;
    for (tag_hash, tag_name, embedding_id, tagged_at) in tags {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        query_builder
            .push("(")
            .push_bind(tag_hash)
            .push(", ")
            .push_bind(tag_name)
            .push(", ")
            .push_bind(embedding_id)
            .push(", ")
            .push_bind(tagged_at)
            .push(")");
    }

    query_builder.push(
        r#" RETURNING
            tag_hash,
            tag_name,
            embedding_id,
            tagged_at;
        "#,
    );

    let rows = query_builder.build().fetch_all(tx).await?;

    let tags = rows
        .into_iter()
        .map(|row| Tag {
            tag_hash: row.get::<i64, _>("tag_hash"),
            tag_name: row.get::<String, _>("tag_name"),
            embedding_id: row.get::<Uuid, _>("embedding_id"),
            tagged_at: row.get::<DateTime<Utc>, _>("tagged_at"),
        })
        .collect();

    Ok(tags)
}

#[cfg(test)]
mod tests_store_tags {
    #[sqlx::test(migrations = false)]
    async fn it_stores_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        todo!()
    }
}
