use crate::{
    chainable::{Chain, ToTx, TxType},
    repositories::embeddings::repository_tx::TxRepository,
};
use chrono::{DateTime, Utc};
use const_fnv1a_hash::fnv1a_hash_str_32;
use futures::future::BoxFuture;
use sqlx::{PgPool, PgTransaction};
use tracing::instrument;
use uuid::Uuid;

pub const EMBEDDING_SIZE: usize = 256;
pub type Value = [f32; 256];

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Tag {
    pub(crate) id: Uuid,
    pub(crate) tag_hash: i64,
    pub(crate) tag_name: String,
    pub(crate) embedding_id: Uuid,
    pub(crate) tagged_at: DateTime<Utc>,
}

impl Tag {
    pub fn new(tag_name: String, embedding_id: Uuid, tagged_at: DateTime<Utc>) -> Self {
        let tag_hash = fnv1a_hash_str_32(&tag_name) as i64;

        Self {
            id: Uuid::now_v7(),
            tag_name,
            tag_hash,
            embedding_id,
            tagged_at,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Embedding {
    pub(super) id: Uuid,
    pub(super) encoded_with: Uuid,
    pub(super) encoded_at: DateTime<Utc>,
    pub(super) value: Value,
}

impl Embedding {
    pub fn new(encoded_with: Uuid, encoded_at: DateTime<Utc>, value: super::Value) -> Self {
        Self {
            id: Uuid::now_v7(),
            encoded_with,
            encoded_at,
            value,
        }
    }

    pub fn id(&self) -> &Uuid {
        &self.id
    }
}

pub struct Similar {
    pub(crate) embedding_id: Uuid,
    pub(crate) distance: f64,
}

impl Similar {
    pub fn embedding_id(&self) -> &Uuid {
        &self.embedding_id
    }

    pub fn distance(&self) -> f64 {
        self.distance
    }
}

pub struct Repository {
    pool: PgPool,
}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn find_similar(
        &self,
        embedding_id: &Uuid,
        tag_name: &str,
        limit: i64,
    ) -> Result<Vec<Similar>, super::Error> {
        super::queries::find_similar(&self.pool, embedding_id, tag_name, limit).await
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
