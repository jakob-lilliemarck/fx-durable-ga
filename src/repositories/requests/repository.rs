use crate::repositories::chainable::{Chain, FromOther, ToTx, TxType};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use sqlx::{
    PgPool, PgTransaction,
    types::chrono::{DateTime, Utc},
};
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Tx error: {0}")]
    Tx(anyhow::Error),
}

#[derive(sqlx::Type, Debug, Clone, PartialEq)]
#[sqlx(type_name = "fx_durable_ga.fitness_goal", rename_all = "lowercase")]
pub enum FitnessGoal {
    Minimize,
    Maximize,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub enum Strategy {
    Rolling {
        /// the maaximum number of evaluations to run before termination
        max_evaluations: u32,
        /// the maximum number of genotypes under evaluation ("active" genotypes)
        population_size: u32,
        /// the number of evaluations to wait before breeding new genotypes. larger pool = more diversity
        selection_interval: u32,
        /// the number of genotypes per tournament
        tournament_size: u32,
        /// the number of genotypes to fetch from the breeding bool during selection
        sample_size: u32,
    },
    Generational {
        max_generations: u32,
        population_size: u32,
    },
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub struct Request {
    pub(crate) id: Uuid,
    pub(crate) requested_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) goal: FitnessGoal,
    pub(crate) threshold: f64,
    pub(crate) strategy: Strategy,
    pub(crate) temperature: f64,
    pub(crate) mutation_rate: f64,
}

pub struct Repository {
    pool: PgPool,
}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash, goal = ?request.goal))]
    pub async fn new_request(&self, request: Request) -> Result<Request, Error> {
        super::queries::new_request(&self.pool, request).await
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %id))]
    pub async fn get_request(&self, id: Uuid) -> Result<Request, Error> {
        super::queries::get_request(&self.pool, id).await
    }
}

impl<'tx> TxType<'tx> for Repository {
    type TxType = TxRepository<'tx>;
    type TxError = Error;
}

impl<'tx> FromOther<'tx> for Repository {
    fn from_other(&self, other: impl ToTx<'tx>) -> Self::TxType {
        TxRepository { tx: other.tx() }
    }
}

impl<'tx> Chain<'tx> for Repository {
    #[instrument(level = "debug", skip(self, f))]
    fn chain<F, R, T>(&'tx self, f: F) -> BoxFuture<'tx, Result<T, Self::TxError>>
    where
        R: ToTx<'tx>,
        F: FnOnce(Self::TxType) -> BoxFuture<'tx, Result<(R, T), anyhow::Error>>
            + Send
            + Sync
            + 'tx,
        T: Send + Sync + 'tx,
    {
        Box::pin(async move {
            let pool = self.pool.clone();
            let tx = pool.begin().await?;

            let (tx, ret) = f(TxRepository { tx }).await.map_err(|err| Error::Tx(err))?;

            let tx: PgTransaction<'_> = tx.tx();
            tx.commit()
                .await
                .map_err(|err| Error::Tx(anyhow::Error::new(err)))?;

            Ok(ret)
        })
    }
}

pub struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    #[instrument(level = "debug", skip(self), fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash, goal = ?request.goal))]
    pub async fn new_request(&mut self, request: Request) -> Result<Request, Error> {
        super::queries::new_request(&mut *self.tx, request).await
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %id))]
    pub async fn get_request(&mut self, id: Uuid) -> Result<Request, Error> {
        super::queries::get_request(&mut *self.tx, id).await
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum RequestValidationError {
    #[error("mutation_rate must be a value between 0.0 and 1.0, got: {0}")]
    InvalidMutationRate(f64),
    #[error("temperature must be a value between 0.0 and 1.0, got: {0}")]
    InvalidTemperature(f64),
}

impl Request {
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash, goal = ?goal, threshold = threshold, temperature = temperature, mutation_rate = mutation_rate))]
    pub(crate) fn new(
        type_name: &str,
        type_hash: i32,
        goal: FitnessGoal,
        threshold: f64,
        strategy: Strategy,
        temperature: f64,
        mutation_rate: f64,
    ) -> Result<Self, RequestValidationError> {
        // Validate temperature (0.0 to 1.0)
        if !(0.0..=1.0).contains(&temperature) {
            return Err(RequestValidationError::InvalidTemperature(temperature));
        }

        // Validate mutation rate (0.0 to 1.0)
        if !(0.0..=1.0).contains(&mutation_rate) {
            return Err(RequestValidationError::InvalidMutationRate(mutation_rate));
        }

        Ok(Self {
            id: Uuid::now_v7(),
            requested_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            goal,
            threshold,
            strategy,
            temperature,
            mutation_rate,
        })
    }

    #[instrument(level = "debug", fields(request_id = %self.id))]
    pub(crate) fn population_size(&self) -> u32 {
        match self.strategy {
            Strategy::Generational {
                population_size, ..
            } => population_size,
            Strategy::Rolling {
                population_size, ..
            } => population_size,
        }
    }

    #[instrument(level = "debug", fields(request_id = %self.id, fitness = fitness, goal = ?self.goal, threshold = self.threshold))]
    pub(crate) fn is_completed(&self, fitness: f64) -> bool {
        match self.goal {
            FitnessGoal::Minimize => fitness <= self.threshold,
            FitnessGoal::Maximize => fitness >= self.threshold,
        }
    }

    #[instrument(level = "debug", fields(request_id = %self.id, evaluations = evaluations))]
    pub fn check_termination(&self, evaluations: i64) -> bool {
        match self.strategy {
            Strategy::Generational {
                max_generations,
                population_size,
            } => evaluations as u32 >= max_generations * population_size,
            Strategy::Rolling {
                max_evaluations, ..
            } => evaluations as u32 >= max_evaluations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::SubsecRound;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request = Request::new(
            "test",
            1,
            FitnessGoal::Maximize,
            0.9,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            0.5,
            0.1,
        )?;
        let request_clone = request.clone();

        let inserted = repository.new_request(request).await?;

        assert_eq!(request_clone.id, inserted.id);
        assert_eq!(
            request_clone.requested_at.trunc_subsecs(6),
            inserted.requested_at
        );
        assert_eq!(request_clone.type_name, inserted.type_name);
        assert_eq!(request_clone.type_hash, inserted.type_hash);
        assert_eq!(request_clone.goal, inserted.goal);
        assert_eq!(request_clone.threshold, inserted.threshold);
        assert_eq!(request_clone.strategy, inserted.strategy);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request = Request::new(
            "test",
            1,
            FitnessGoal::Maximize,
            0.9,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            0.5,
            0.1,
        )?;
        let request_clone = request.clone();

        let _ = repository.new_request(request).await?;
        let inserted = repository.new_request(request_clone).await;

        assert!(inserted.is_err());

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request = Request::new(
            "test",
            1,
            FitnessGoal::Maximize,
            0.9,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            0.5,
            0.1,
        )?;
        let request_id = request.id;

        let _ = repository.new_request(request).await?;
        let selected = repository.get_request(request_id).await?;

        assert_eq!(request_id, selected.id);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request = Request::new(
            "test",
            1,
            FitnessGoal::Maximize,
            0.9,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            0.5,
            0.1,
        )?;
        let request_id = request.id;

        let selected = repository.get_request(request_id).await;

        assert!(selected.is_err());
        Ok(())
    }
}
