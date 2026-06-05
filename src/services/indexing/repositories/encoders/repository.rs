use super::{Digest, Encoder, EncoderAvailability};
use crate::infrastructure::db::{self, ReadPool, WritePool};
use chrono::{DateTime, Utc};
use moka::sync::Cache;
use sqlx::PgTransaction;
use std::sync::RwLock;
use std::{sync::Arc, time::Duration};
use tracing::instrument;

/// In-memory cache for encoder instances.
pub struct EncoderCache {
    cache: Cache<Digest, Arc<super::Encoder>>,
}

impl EncoderCache {
    #[instrument(level = "debug")]
    pub fn new(ttl: Duration, capacity: usize) -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(capacity as u64)
                .time_to_live(ttl)
                .build(),
        }
    }

    pub fn get(&self, encoder_id: &Digest) -> Option<Arc<super::Encoder>> {
        self.cache.get(encoder_id)
    }

    pub fn set(&mut self, encoder: Arc<super::Encoder>) {
        self.cache.insert(*encoder.digest(), encoder);
    }
}

/// Encoders read repository, exposing read-only queries
#[derive(Clone)]
pub struct Read {
    ro: ReadPool,
    cache: Arc<RwLock<EncoderCache>>,
}

/// Encoders write repository, holding a write pool for transactions
#[derive(Debug, Clone)]
pub struct Write {
    wr: WritePool,
}

/// Encoders write repository, scoped to an active transaction
pub struct WriteTx<'tx> {
    tx: &'tx mut PgTransaction<'static>,
}

impl db::Tx for Write {
    type Error = super::Error;

    fn tx(self) -> db::TxFut<Self::Error> {
        let pool = self.wr.pool.clone();
        Box::pin(async move {
            let tx = pool.begin().await?;
            Ok(tx)
        })
    }
}

impl Read {
    pub fn new(ro: ReadPool, cache: Arc<RwLock<EncoderCache>>) -> Self {
        Self { ro, cache }
    }

    #[instrument(level = "debug", skip(self))]
    pub async fn get_encoder(
        &self,
        indexer_id: &Digest,
    ) -> Result<Arc<super::Encoder>, super::Error> {
        if let Some(encoder) = self.cache.read()?.get(indexer_id) {
            return Ok(encoder);
        }

        if let Some(encoder) = super::queries::get_encoder(&self.ro.pool, indexer_id).await? {
            let encoder = Arc::new(encoder);
            self.cache.write()?.set(encoder.clone());
            return Ok(encoder);
        }

        Err(super::Error::NotFound(indexer_id.clone()))
    }

    /// Returns digests of encoders that are available from the given list.
    #[instrument(level = "debug", skip(self))]
    pub async fn get_enabled_encoder_digests(
        &self,
        encoder_digests: &[Digest],
    ) -> Result<Vec<super::Digest>, super::Error> {
        super::queries::get_available_encoder_digests(&self.ro.pool, encoder_digests).await
    }

    /// Returns the availability status of the given encoder.
    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn get_encoder_availability(
        &self,
        indexer_id: &Digest,
    ) -> Result<EncoderAvailability, super::Error> {
        super::queries::get_encoder_availability(&self.ro.pool, indexer_id).await
    }
}

impl Write {
    pub fn new(wr: WritePool) -> Self {
        Self { wr }
    }
}

impl<'tx> WriteTx<'tx> {
    #[instrument(level = "debug", skip(tx))]
    pub fn new(tx: &'tx mut PgTransaction<'static>) -> Self {
        Self { tx }
    }

    /// Stores a trained encoder within the current transaction.
    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn store_encoder(
        &mut self,
        encoder: &Encoder,
        trained_at: &DateTime<Utc>,
    ) -> Result<Encoder, super::Error> {
        super::queries::store_encoder(&mut **self.tx, encoder, trained_at).await
    }

    /// Updates encoder availability within the current transaction.
    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn store_encoder_availability(
        &mut self,
        indexer_id: &Digest,
        is_available: bool,
    ) -> Result<(EncoderAvailability, bool), super::Error> {
        let revised_at = Utc::now();

        let result = super::queries::store_encoder_availability(
            &mut **self.tx,
            indexer_id,
            is_available,
            &revised_at,
        )
        .await?;

        Ok(result)
    }
}
