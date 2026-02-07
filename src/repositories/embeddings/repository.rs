use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use sqlx::{PgExecutor, PgPool, PgTransaction};
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

use crate::{
    chainable::{Chain, ToTx, TxType},
    repositories::embeddings::queries::TagRow,
};

const SIZE: usize = 256;

pub type Value = [f32; 256];

pub struct Tag {
    pub(crate) tag_hash: i64,
    pub(crate) tag_name: String,
    pub(crate) embedding_id: Uuid,
    pub(crate) tagged_at: DateTime<Utc>,
}

pub struct Embedding {
    pub(crate) id: Uuid,
    pub(crate) encoded_with: Uuid,
    pub(crate) encoded_at: DateTime<Utc>,
    pub(crate) value: Value,
}

pub struct Similar {
    pub(crate) id: Uuid,
    pub(crate) distance: f64,
}

pub trait Executable<'tx> {
    fn executor(&'tx mut self) -> impl PgExecutor<'tx>;
}

pub struct Repository {
    pool: PgPool,
}

impl<'tx> Executable<'tx> for Repository {
    fn executor(&'tx mut self) -> impl PgExecutor<'tx> {
        &self.pool
    }
}

impl<'tx> Executable<'tx> for Arc<Repository> {
    fn executor(&'tx mut self) -> impl PgExecutor<'tx> {
        &(**self).pool
    }
}

pub struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl<'tx> Executable<'tx> for TxRepository<'tx> {
    fn executor(&'tx mut self) -> impl PgExecutor<'tx> {
        &mut *self.tx
    }
}

pub trait EmbeddingsRepository<'tx>: Executable<'tx> {
    /// Similarity search at the database level
    async fn get_similar(
        &'tx mut self,
        id: &Uuid,
        tags: &[i64],
        limit: i64,
    ) -> Result<Vec<Similar>, super::Error> {
        super::queries::get_similar(self.executor(), id, tags, limit).await
    }

    /// Inserts if the embedding does not exists, otherwise overwrites the current vale.
    async fn store_embedding(
        &'tx mut self,
        embedding: &Embedding,
    ) -> Result<Embedding, super::Error> {
        super::queries::store_embedding(self.executor(), embedding).await
    }

    async fn store_tags<I>(&'tx mut self, tags: I) -> Result<Vec<Tag>, super::Error>
    where
        I: IntoIterator<Item = TagRow<'tx>>,
    {
        super::queries::store_tags(self.executor(), tags).await
    }
}

impl<'tx> EmbeddingsRepository<'tx> for Repository {}
impl<'tx> EmbeddingsRepository<'tx> for TxRepository<'tx> {}
impl<'tx> EmbeddingsRepository<'tx> for Arc<Repository> {}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    /// Extracts the underlying database transaction.
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
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
