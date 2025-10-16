use super::{Error, TxRepository};
use crate::models::Genotype;
use crate::repositories::chainable::{Chain, ToTx, TxType};
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

    #[instrument(level = "debug", skip(self), fields(genotype_id = %id))]
    pub(crate) async fn get_genotype(&self, id: &Uuid) -> Result<Genotype, Error> {
        super::queries::get_genotype(&self.pool, id).await
    }

    #[instrument(level = "debug", skip(self), fields(genotype_id = ?ids))]
    pub(crate) async fn get_genotypes(&self, ids: &[Uuid]) -> Result<Vec<Genotype>, Error> {
        super::queries::get_genotypes(&self.pool, ids).await
    }
}

impl<'tx> TxType<'tx> for Repository {
    type TxType = TxRepository<'tx>;
    type TxError = Error;
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
