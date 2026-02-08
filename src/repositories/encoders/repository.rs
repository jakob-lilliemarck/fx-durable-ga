use crate::{
    chainable::{Chain, ToTx, TxType},
    repositories::encoders::repository_tx::TxRepository,
};
use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use sqlx::{PgPool, PgTransaction};
use tracing::instrument;
use uuid::Uuid;

/// An encoder model configuration
pub struct Encoder {
    pub(crate) id: Uuid,
    pub(crate) model_type: String,
    pub(crate) model_config: serde_json::Value,
    pub(crate) model_weights: Vec<u8>,
    pub(crate) model_format: String,
    pub(crate) shape_in: Vec<i32>,
    pub(crate) shape_out: i32,
    pub(crate) trained_at: DateTime<Utc>,
    pub(crate) trained_on_checksum: Vec<u8>,
}

impl Encoder {
    pub fn id(&self) -> Uuid {
        self.id
    }
}

pub struct Repository {
    pool: PgPool,
}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_encoder(&self, id: &Uuid) -> Result<Option<Encoder>, super::Error> {
        super::queries::get_encoder(&self.pool, id).await
    }
}

impl<'tx> TxType<'tx> for Repository {
    type TxType = TxRepository<'tx>;
    type TxError = super::Error;
}

impl<'tx> Chain<'tx> for Repository {
    /// Executes a function within a database transaction.
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

            let (tx, ret) = f(TxRepository { tx })
                .await
                .map_err(|err| super::Error::Tx(err))?;

            let tx: PgTransaction<'_> = tx.tx();
            tx.commit()
                .await
                .map_err(|err| super::Error::Tx(anyhow::Error::new(err)))?;

            Ok(ret)
        })
    }
}
