use super::Error;
use super::repository_tx::TxRepository;
use crate::models::{Crossover, FitnessGoal, Mutagen, Request, Strategy};
use crate::repositories::chainable::{Chain, FromOther, ToTx, TxType};
use futures::future::BoxFuture;
use sqlx::{PgPool, PgTransaction, types::chrono::Utc};
use tracing::instrument;
use uuid::Uuid;

pub(crate) struct Repository {
    pool: PgPool,
}

impl Repository {
    pub(crate) fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %id))]
    pub(crate) async fn get_request(&self, id: Uuid) -> Result<Request, Error> {
        super::queries::get_request(&self.pool, &id).await
    }
}

impl<'tx> TxType<'tx> for Repository {
    type TxType = TxRepository<'tx>;
    type TxError = Error;
}

impl<'tx> FromOther<'tx> for Repository {
    fn from_other(&self, other: impl ToTx<'tx>) -> Self::TxType {
        TxRepository::new(other.tx())
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

            let (tx, ret) = f(TxRepository::new(tx))
                .await
                .map_err(|err| Error::Tx(err))?;

            let tx: PgTransaction<'_> = tx.tx();
            tx.commit()
                .await
                .map_err(|err| Error::Tx(anyhow::Error::new(err)))?;

            Ok(ret)
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RequestValidationError {}

impl Request {
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash, goal = ?goal, threshold = threshold, mutagen = ?mutagen))]
    pub(crate) fn new(
        type_name: &str,
        type_hash: i32,
        goal: FitnessGoal,
        threshold: f64,
        strategy: Strategy,
        mutagen: Mutagen,
        crossover: Crossover,
    ) -> Result<Self, RequestValidationError> {
        Ok(Self {
            id: Uuid::now_v7(),
            requested_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            goal,
            threshold,
            strategy,
            mutagen,
            crossover,
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
}
