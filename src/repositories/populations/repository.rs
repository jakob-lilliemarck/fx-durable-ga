use crate::repositories::chainable::{Chain, FromOther, FromTx, ToTx, TxType};
use sqlx::{PgPool, PgTransaction};
use std::future::Future;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Tx error: {0}")]
    Tx(anyhow::Error),
}

#[derive(Clone)]
pub struct Repository {
    pool: PgPool,
}

impl Repository {
    pub(crate) fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    #[instrument(level = "debug", skip(self), fields(individuals_count = individuals.len()))]
    pub(crate) fn add_to_population(
        &self,
        individuals: &[(Uuid, Uuid)],
    ) -> impl Future<Output = Result<(), Error>> {
        super::queries::add_to_population(&self.pool, individuals)
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub(crate) fn remove_from_population(
        &self,
        request_id: &Uuid,
        genotype_id: &Uuid,
    ) -> impl Future<Output = Result<(), Error>> {
        super::queries::remove_from_population(&self.pool, request_id, genotype_id)
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) fn get_population_count(
        &self,
        request_id: &Uuid,
    ) -> impl Future<Output = Result<i64, Error>> {
        super::queries::get_population_count(&self.pool, request_id)
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
    fn chain<F, R, T>(&'tx self, f: F) -> futures::future::BoxFuture<'tx, Result<T, Self::TxError>>
    where
        R: ToTx<'tx>,
        F: FnOnce(Self::TxType) -> futures::future::BoxFuture<'tx, Result<(R, T), anyhow::Error>>
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
    #[instrument(level = "debug", skip(self), fields(individuals_count = individuals.len()))]
    pub(crate) fn add_to_population(
        &mut self,
        individuals: &[(Uuid, Uuid)],
    ) -> impl Future<Output = Result<(), Error>> {
        super::queries::add_to_population(&mut *self.tx, individuals)
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub(crate) fn remove_from_population(
        &mut self,
        request_id: &Uuid,
        genotype_id: &Uuid,
    ) -> impl Future<Output = Result<(), Error>> {
        super::queries::remove_from_population(&mut *self.tx, request_id, genotype_id)
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) fn get_population_count(
        &mut self,
        request_id: &Uuid,
    ) -> impl Future<Output = Result<i64, Error>> {
        super::queries::get_population_count(&mut *self.tx, request_id)
    }
}

impl<'tx> FromTx<'tx> for TxRepository<'tx> {
    fn from_tx(other: impl ToTx<'tx>) -> Self {
        TxRepository { tx: other.tx() }
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
