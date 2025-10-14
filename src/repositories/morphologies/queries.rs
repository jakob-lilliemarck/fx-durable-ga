use super::{Error, Morphology};
use crate::repositories::morphologies::GeneBounds;
use chrono::{DateTime, Utc};
use sqlx::PgExecutor;
use tracing::instrument;

#[derive(Debug)]
struct DBMorphology {
    type_name: String,
    type_hash: i32,
    revised_at: DateTime<Utc>,
    gene_bounds: serde_json::Value,
}

impl TryFrom<DBMorphology> for Morphology {
    type Error = Error;

    #[instrument(level = "debug", fields(type_name = %db_morphology.type_name, type_hash = db_morphology.type_hash))]
    fn try_from(db_morphology: DBMorphology) -> Result<Self, Self::Error> {
        let gene_bounds: Vec<GeneBounds> = serde_json::from_value(db_morphology.gene_bounds)?;

        Ok(Morphology {
            type_hash: db_morphology.type_hash,
            type_name: db_morphology.type_name,
            revised_at: db_morphology.revised_at,
            gene_bounds: gene_bounds,
        })
    }
}

impl TryFrom<Morphology> for DBMorphology {
    type Error = Error;

    #[instrument(level = "debug", fields(type_name = %morphology.type_name, type_hash = morphology.type_hash))]
    fn try_from(morphology: Morphology) -> Result<Self, Self::Error> {
        let gene_bounds_json = serde_json::to_value(&morphology.gene_bounds)?;

        Ok(DBMorphology {
            type_hash: morphology.type_hash,
            type_name: morphology.type_name,
            revised_at: morphology.revised_at,
            gene_bounds: gene_bounds_json,
        })
    }
}

#[instrument(level = "debug", skip(tx), fields(type_name = %morphology.type_name, type_hash = morphology.type_hash))]
pub async fn new_morphology<'tx, E: PgExecutor<'tx>>(
    tx: E,
    morphology: Morphology,
) -> Result<Morphology, Error> {
    let db_morphology = DBMorphology::try_from(morphology)?;
    let db_morphology = sqlx::query_as!(
        DBMorphology,
        r#"
            INSERT INTO fx_durable_ga.morphologies (
                revised_at,
                type_name,
                type_hash,
                gene_bounds
            )
            VALUES ($1, $2, $3, $4)
            RETURNING
                revised_at,
                type_name,
                type_hash,
                gene_bounds
            "#,
        db_morphology.revised_at,
        db_morphology.type_name,
        db_morphology.type_hash,
        db_morphology.gene_bounds
    )
    .fetch_one(tx)
    .await?;

    let morphology = Morphology::try_from(db_morphology)?;
    Ok(morphology)
}

#[instrument(level = "debug", skip(tx), fields(type_hash = type_hash))]
pub async fn get_morphology<'tx, E: PgExecutor<'tx>>(
    tx: E,
    type_hash: i32,
) -> Result<Morphology, Error> {
    let db_morphology = sqlx::query_as!(
        DBMorphology,
        r#"
            SELECT
                type_name,
                type_hash,
                revised_at,
                gene_bounds
            FROM fx_durable_ga.morphologies
            WHERE type_hash = $1
        "#,
        type_hash
    )
    .fetch_one(tx)
    .await
    .map_err(|err| match err {
        sqlx::Error::RowNotFound => Error::NotFound,
        err => Error::Database(err),
    })?;

    let morphology = Morphology::try_from(db_morphology)?;
    Ok(morphology)
}
