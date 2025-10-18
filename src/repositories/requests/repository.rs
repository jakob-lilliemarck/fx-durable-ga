use super::Error;
use super::repository_tx::TxRepository;
use crate::models::{Request, RequestConclusion};
use crate::repositories::chainable::{Chain, FromOther, ToTx, TxType};
use futures::future::BoxFuture;
use sqlx::{PgPool, PgTransaction};
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

    #[instrument(level = "debug", skip(self), fields(request_id = %id))]
    pub(crate) fn get_request_conclusion(
        &self,
        id: &Uuid,
    ) -> impl Future<Output = Result<Option<RequestConclusion>, Error>> {
        super::queries::get_request_conclusion(&self.pool, id)
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_conclusion.request_id, concluded_at = %request_conclusion.concluded_at, concluded_with = ?request_conclusion.concluded_with))]
    pub(crate) fn new_request_conclusion(
        &self,
        request_conclusion: &RequestConclusion,
    ) -> impl Future<Output = Result<RequestConclusion, Error>> {
        super::queries::new_request_conclusion(&self.pool, request_conclusion)
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
