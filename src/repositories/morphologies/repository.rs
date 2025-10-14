use crate::repositories::genotypes;
use chrono::{DateTime, Utc};
use rand::{Rng, rngs::ThreadRng};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::instrument;

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

    #[instrument(level = "debug", skip(self), fields(type_name = %morphology.type_name, type_hash = morphology.type_hash))]
    pub async fn new_morphology(&self, morphology: Morphology) -> Result<Morphology, Error> {
        super::queries::new_morphology(&self.pool, morphology).await
    }

    #[instrument(level = "debug", skip(self), fields(type_hash = type_hash))]
    pub async fn get_morphology(&self, type_hash: i32) -> Result<Morphology, Error> {
        super::queries::get_morphology(&self.pool, type_hash).await
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct GeneBounds {
    pub(crate) lower: i32,
    pub(crate) upper: i32,
    pub(crate) divisor: i32,
}

impl GeneBounds {
    #[instrument(level = "debug", fields(lower = lower, upper = upper, divisor = divisor))]
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

    #[instrument(level = "debug", skip(rng), fields(lower = self.lower, upper = self.upper, divisor = self.divisor))]
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
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash, gene_bounds_count = gene_bounds.len()))]
    pub fn new(type_name: &str, type_hash: i32, gene_bounds: Vec<GeneBounds>) -> Self {
        Self {
            revised_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            gene_bounds,
        }
    }

    #[instrument(level = "debug", fields(type_name = %self.type_name, type_hash = self.type_hash, gene_bounds_count = self.gene_bounds.len()))]
    pub fn random(&self) -> Vec<genotypes::Gene> {
        let mut rng = rand::rng();

        self.gene_bounds
            .iter()
            .map(|gene_bound| gene_bound.random(&mut rng))
            .collect()
    }
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
