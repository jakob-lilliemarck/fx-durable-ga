use super::Error;
use super::repository_tx::TxRepository;
use crate::{
    models::Population,
    repositories::chainable::{Chain, FromOther, ToTx, TxType},
};
use sqlx::{PgPool, PgTransaction};
use std::future::Future;
use tracing::instrument;
use uuid::Uuid;

#[derive(Clone)]
pub(crate) struct Repository {
    pool: PgPool,
}

impl Repository {
    pub(crate) fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) fn get_population(
        &self,
        request_id: &Uuid,
    ) -> impl Future<Output = Result<Population, Error>> {
        super::queries::get_population(&self.pool, request_id)
    }

    #[instrument(level = "debug", skip(self), fields(filter = ?filter))]
    pub(crate) fn search_individuals(
        &self,
        filter: &super::queries::Filter,
        limit: i64,
    ) -> impl Future<Output = Result<Vec<(Uuid, Option<f64>)>, Error>> {
        super::queries::search_individuals(&self.pool, filter, limit)
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
