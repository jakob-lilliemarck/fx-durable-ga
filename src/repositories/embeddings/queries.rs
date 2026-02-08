use crate::repositories::embeddings::{
    Embedding,
    repository::{EMBEDDING_SIZE, Similar, Tag},
};
use chrono::{DateTime, Utc};
use const_fnv1a_hash::fnv1a_hash_str_32;
use pgvector::Vector;
use sqlx::{PgExecutor, Row};
use uuid::Uuid;

pub(crate) async fn find_similar<'tx, E: PgExecutor<'tx>>(
    tx: E,
    embedding_id: &Uuid,
    tag_name: &str,
    limit: i64,
) -> Result<Vec<Similar>, super::Error> {
    let tag_hash = fnv1a_hash_str_32(tag_name) as i64;

    let similar = sqlx::query_as!(
        Similar,
        r#"
        WITH query_embedding AS (
            SELECT value
            FROM embeddings
            WHERE id = $1
        )
        SELECT
            te.id AS "embedding_id!: Uuid",
            te.value <=> qe.value AS "distance!: f64"
        FROM tagged_embeddings te
        CROSS JOIN query_embedding qe
        WHERE
            te.id != $1
            AND te.tag_hash = $2
        ORDER BY "distance!: f64", te.id
        LIMIT $3;
        "#,
        embedding_id,
        tag_hash,
        limit
    )
    .fetch_all(tx)
    .await?;

    Ok(similar)
}

#[cfg(test)]
mod tests_get_similar {
    use crate::{
        chainable::Chain,
        repositories::embeddings::{Embedding, Repository, Tag, repository::EMBEDDING_SIZE},
    };
    use chrono::SubsecRound;
    use chrono::Utc;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_gets_similar_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let repository = Repository::new(pool);

        let now = Utc::now().trunc_subsecs(6);

        let embeddings = &[
            Embedding {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
                encoded_with: Uuid::nil(),
                encoded_at: now,
                value: [1.1_f32; EMBEDDING_SIZE],
            },
            Embedding {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
                encoded_with: Uuid::nil(),
                encoded_at: now,
                value: [1.111_f32; EMBEDDING_SIZE],
            },
            Embedding {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap(),
                encoded_with: Uuid::nil(),
                encoded_at: now,
                value: [1.11_f32; EMBEDDING_SIZE],
            },
            Embedding {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap(),
                encoded_with: Uuid::nil(),
                encoded_at: now,
                value: [1.0_f32; EMBEDDING_SIZE],
            },
        ];

        let tags: Vec<Tag> = embeddings
            .iter()
            .map(|e| Tag::new("test_similarity".to_string(), e.id, now))
            .collect();

        repository
            .chain(|mut tx| {
                let embeddings = embeddings.clone();

                Box::pin(async move {
                    tx.store_embeddings(&embeddings).await?;
                    tx.store_tags(&tags).await?;
                    Ok((tx, ()))
                })
            })
            .await?;

        let similar = repository
            .find_similar(&embeddings[3].id, "test_similarity", 2)
            .await?;

        assert_eq!(similar[0].embedding_id, embeddings[0].id);
        assert_eq!(similar[1].embedding_id, embeddings[2].id);

        Ok(())
    }
}

pub(crate) async fn store_embeddings<'tx, 'a, I, E>(
    tx: E,
    embeddings: I,
) -> Result<Vec<Embedding>, super::Error>
where
    I: IntoIterator<Item = &'a Embedding>,
    E: PgExecutor<'tx>,
{
    let mut embeddings = embeddings.into_iter().peekable();
    if embeddings.peek().is_none() {
        return Ok(vec![]);
    }

    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.embeddings (
            id,
            encoded_at,
            encoded_with,
            value
        )
        VALUES ",
    );

    let mut first = true;
    for e in embeddings {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        let value = Vector::from(e.value.to_vec());
        query_builder
            .push("(")
            .push_bind(e.id)
            .push(", ")
            .push_bind(e.encoded_at)
            .push(", ")
            .push_bind(e.encoded_with)
            .push(", ")
            .push_bind(value)
            .push(")");
    }

    query_builder.push(" RETURNING id, encoded_at, encoded_with, value;");

    let query = query_builder.build();
    let rows = query.fetch_all(tx).await?;

    rows.into_iter()
        .map(|row| {
            let vector = row.get::<Vector, _>("value");
            let value: [f32; EMBEDDING_SIZE] = vector
                .to_vec()
                .try_into()
                .map_err(|v: Vec<f32>| super::Error::InvalidDimension(v.len()))?;

            Ok(Embedding {
                id: row.get::<Uuid, _>("id"),
                encoded_at: row.get::<DateTime<Utc>, _>("encoded_at"),
                encoded_with: row.get::<Uuid, _>("encoded_with"),
                value: value,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests_store_embeddings {
    use crate::{
        chainable::Chain,
        repositories::embeddings::{Embedding, Repository, repository::EMBEDDING_SIZE},
    };
    use chrono::SubsecRound;
    use chrono::Utc;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_stores_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let repository = Repository::new(pool);

        let embeddings = &[Embedding {
            id: Uuid::nil(),
            encoded_with: Uuid::nil(),
            encoded_at: Utc::now().trunc_subsecs(6),
            value: [1_f32; EMBEDDING_SIZE],
        }];

        let stored: Vec<Embedding> = repository
            .chain(|mut tx| {
                let embeddings = embeddings.clone();
                Box::pin(async move {
                    let stored = tx.store_embeddings(&embeddings).await?;
                    Ok((tx, stored))
                })
            })
            .await?;

        for (e, s) in embeddings.iter().zip(stored) {
            assert_eq!(e.id, s.id);
            assert_eq!(e.encoded_at, s.encoded_at);
            assert_eq!(e.encoded_with, s.encoded_with);
            assert_eq!(e.value, s.value);
        }

        Ok(())
    }
}

pub(crate) async fn store_tags<'tx, 'a, I, E>(tx: E, tags: I) -> Result<Vec<Tag>, super::Error>
where
    I: IntoIterator<Item = &'a Tag>,
    E: PgExecutor<'tx>,
{
    let mut tags = tags.into_iter().peekable();
    if tags.peek().is_none() {
        return Ok(vec![]);
    }

    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.embedding_tags (
            id,
            tag_hash,
            tag_name,
            embedding_id,
            tagged_at
        )
        VALUES ",
    );

    let mut first = true;
    for t in tags {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        query_builder
            .push("(")
            .push_bind(t.id)
            .push(", ")
            .push_bind(t.tag_hash)
            .push(", ")
            .push_bind(&t.tag_name)
            .push(", ")
            .push_bind(t.embedding_id)
            .push(", ")
            .push_bind(t.tagged_at)
            .push(")");
    }

    query_builder.push(
        r#" RETURNING
            id,
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
            id: row.get::<Uuid, _>("id"),
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
    use crate::{
        chainable::Chain,
        repositories::embeddings::{Embedding, Repository, Tag, repository::EMBEDDING_SIZE},
    };
    use chrono::SubsecRound;
    use chrono::Utc;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_stores_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let repository = Repository::new(pool);

        let now = Utc::now().trunc_subsecs(6);

        let embeddings = &[Embedding {
            id: Uuid::nil(),
            encoded_with: Uuid::nil(),
            encoded_at: now,
            value: [1_f32; EMBEDDING_SIZE],
        }];

        let tags = &[
            Tag::new("tag_1".to_string(), embeddings[0].id, now),
            Tag::new("tag_2".to_string(), embeddings[0].id, now),
            Tag::new("tag_3".to_string(), embeddings[0].id, now),
        ];

        let stored = repository
            .chain(|mut tx| {
                let embeddings = embeddings.clone();
                let tags = tags.clone();
                Box::pin(async move {
                    tx.store_embeddings(&embeddings).await?;
                    let tags = tx.store_tags(&tags).await?;
                    Ok((tx, tags))
                })
            })
            .await?;

        for (t, s) in tags.iter().zip(stored) {
            assert_eq!(t.id, s.id);
            assert_eq!(t.tag_name, s.tag_name);
            assert_eq!(t.tag_hash, s.tag_hash);
            assert_eq!(t.embedding_id, s.embedding_id);
            assert_eq!(t.tagged_at, s.tagged_at);
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_error_on_duplicate_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let repository = Repository::new(pool);

        let now = Utc::now().trunc_subsecs(6);

        let embeddings = &[Embedding {
            id: Uuid::nil(),
            encoded_with: Uuid::nil(),
            encoded_at: now,
            value: [1_f32; EMBEDDING_SIZE],
        }];

        let tags = &[
            Tag::new("duplicate".to_string(), embeddings[0].id, now),
            Tag::new("duplicate".to_string(), embeddings[0].id, now),
        ];

        let result = repository
            .chain(|mut tx| {
                let embeddings = embeddings.clone();
                let tags = tags.clone();
                Box::pin(async move {
                    tx.store_embeddings(&embeddings).await?;
                    let tags = tx.store_tags(&tags).await?;
                    Ok((tx, tags))
                })
            })
            .await;

        assert!(result.is_err());

        Ok(())
    }
}
