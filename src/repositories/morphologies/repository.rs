use crate::repositories::genotypes;
use chrono::{DateTime, Utc};
use rand::{Rng, rngs::ThreadRng};
use serde::{Deserialize, Serialize};
use sqlx::{PgExecutor, PgPool};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Not found")]
    NotFound,
}

pub struct Repository {
    pool: PgPool,
}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn new_morphology(&self, morphology: Morphology) -> Result<Morphology, Error> {
        new_morphology(&self.pool, morphology).await
    }

    pub async fn get_morphology(&self, type_hash: i32) -> Result<Morphology, Error> {
        get_morphology(&self.pool, type_hash).await
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GeneBoundError {
    #[error(
        "InvalidBounds: lower bound must be smaller than upper. lower = {lower}, upper={upper}"
    )]
    InvalidBound { lower: i32, upper: i32 },
    #[error("DivisorOverflow: divisor is too large. divisor={divisor}, max={max}")]
    DivisorOverflow { divisor: u32, max: i32 },
}

impl GeneBoundError {
    fn divisor_overflow(divisor: u32) -> Self {
        Self::DivisorOverflow {
            divisor,
            max: i32::MAX,
        }
    }

    fn invalid_bound(lower: i32, upper: i32) -> Self {
        Self::InvalidBound { lower, upper }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct GeneBounds {
    pub(crate) lower: i32,
    pub(crate) upper: i32,
    pub(crate) divisor: i32,
}

impl GeneBounds {
    pub fn new(lower: i32, upper: i32, divisor: u32) -> Result<Self, GeneBoundError> {
        if lower > upper {
            return Err(GeneBoundError::invalid_bound(lower, upper));
        };

        let divisor =
            i32::try_from(divisor).map_err(|_| GeneBoundError::divisor_overflow(divisor))?;

        Ok(Self {
            lower,
            upper,
            divisor,
        })
    }

    pub fn random(&self, rng: &mut ThreadRng) -> genotypes::Gene {
        rng.random_range(0..self.divisor as i64)
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct Morphology {
    pub(crate) revised_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) gene_bounds: Vec<GeneBounds>,
}

impl Morphology {
    pub fn new(type_name: &str, type_hash: i32, gene_bounds: Vec<GeneBounds>) -> Self {
        Self {
            revised_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            gene_bounds,
        }
    }

    pub fn random(&self, rng: &mut ThreadRng) -> Vec<genotypes::Gene> {
        self.gene_bounds
            .iter()
            .map(|gene_bound| gene_bound.random(rng))
            .collect()
    }
}

struct DBMorphology {
    type_name: String,
    type_hash: i32,
    revised_at: DateTime<Utc>,
    gene_bounds: serde_json::Value,
}

impl TryFrom<DBMorphology> for Morphology {
    type Error = Error;
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

async fn new_morphology<'tx, E: PgExecutor<'tx>>(
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

async fn get_morphology<'tx, E: PgExecutor<'tx>>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::SubsecRound;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_morphology(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
            ],
        );
        let morphology_clone = morphology.clone();

        let inserted = repository.new_morphology(morphology).await?;

        assert_eq!(
            morphology_clone.revised_at.trunc_subsecs(6),
            inserted.revised_at
        );
        assert_eq!(morphology_clone.type_name, inserted.type_name);
        assert_eq!(morphology_clone.type_hash, inserted.type_hash);
        assert_eq!(morphology_clone.gene_bounds, inserted.gene_bounds);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
            ],
        );
        let morphology_clone = morphology.clone();

        let _ = repository.new_morphology(morphology).await?;
        let inserted = repository.new_morphology(morphology_clone).await;

        assert!(inserted.is_err());

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_morphology(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
            ],
        );
        let morphology_type_hash = morphology.type_hash;

        let _ = repository.new_morphology(morphology).await?;
        let selected = repository.get_morphology(morphology_type_hash).await?;

        assert_eq!(morphology_type_hash, selected.type_hash);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
            ],
        );
        let morphology_type_hash = morphology.type_hash;

        let selected = repository.get_morphology(morphology_type_hash).await;

        assert!(selected.is_err());
        assert!(matches!(selected, Err(Error::NotFound)));

        Ok(())
    }
}
