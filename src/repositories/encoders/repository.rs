use std::{collections::HashMap, sync::Arc, time::Duration};

use crate::{
    chainable::{Chain, ToTx, TxType},
    repositories::encoders::repository_tx::TxRepository,
};
use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use moka::sync::Cache;
use sqlx::{PgPool, PgTransaction};
use std::sync::RwLock;
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

/// A relation between an encoder and a foreign type
pub struct EncoderPairing {
    pub(super) encoder_id: Uuid,
    pub(super) type_hash: i32,
}

impl EncoderPairing {
    /// Group by encoder_id -> list of type_hashes
    pub fn group_by_encoder(pairings: Vec<EncoderPairing>) -> HashMap<Uuid, Vec<i32>> {
        let mut map: HashMap<Uuid, Vec<i32>> = HashMap::new();
        for pairing in pairings {
            map.entry(pairing.encoder_id)
                .or_insert_with(Vec::new)
                .push(pairing.type_hash);
        }
        map
    }

    /// Group by type_hash -> list of encoder_ids
    pub fn group_by_type_hash(pairings: Vec<EncoderPairing>) -> HashMap<i32, Vec<Uuid>> {
        let mut map: HashMap<i32, Vec<Uuid>> = HashMap::new();
        for pairing in pairings {
            map.entry(pairing.type_hash)
                .or_insert_with(Vec::new)
                .push(pairing.encoder_id);
        }
        map
    }
}

impl Encoder {
    pub fn id(&self) -> Uuid {
        self.id
    }
}

pub struct EncoderCache {
    cache: Cache<Uuid, Arc<Encoder>>,
}

impl EncoderCache {
    pub fn new(ttl: Duration, capacity: usize) -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(capacity as u64)
                .time_to_live(ttl)
                .build(),
        }
    }

    pub fn get(&self, encoder_id: &Uuid) -> Option<Arc<Encoder>> {
        self.cache.get(encoder_id)
    }

    pub fn set(&mut self, encoder: Arc<Encoder>) {
        self.cache.insert(encoder.id, encoder);
    }
}

pub struct Repository {
    pool: PgPool,
    cache: RwLock<EncoderCache>,
}

impl Repository {
    pub fn new(pool: PgPool, ttl: Duration, capacity: usize) -> Self {
        Self {
            pool,
            cache: RwLock::new(EncoderCache::new(ttl, capacity)),
        }
    }

    pub async fn get_encoder(&self, encoder_id: &Uuid) -> Result<Arc<Encoder>, super::Error> {
        if let Some(encoder) = self.cache.read()?.get(encoder_id) {
            return Ok(encoder);
        }

        if let Some(encoder) = super::queries::get_encoder(&self.pool, encoder_id).await? {
            let encoder = Arc::new(encoder);
            self.cache.write()?.set(encoder.clone());
            return Ok(encoder);
        }

        Err(super::Error::NotFound(encoder_id.to_owned()))
    }

    pub async fn get_encoder_pairings(
        &self,
        type_hashes: &[i32],
    ) -> Result<Vec<EncoderPairing>, super::Error> {
        super::queries::get_encoder_pairings(&self.pool, type_hashes).await
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
