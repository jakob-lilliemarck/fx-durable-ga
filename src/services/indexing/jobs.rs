use fx_mq_jobs::Message;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tracing::instrument;
use uuid::Uuid;

// ============================================================
// IndexGenotype
// ============================================================

/// Message to trigger indexing batch job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexGenotypesMessage {
    pub genotype_ids: Vec<Uuid>,
    pub encoder_id: Uuid,
    pub embedding_tags: Vec<String>,
}

impl fx_mq_jobs::Message for IndexGenotypesMessage {
    const NAME: &str = "IndexGenotypes";
}

/// Handler that processes indexing batch-jobs
pub struct IndexGenotypesHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for IndexGenotypesHandler {
    type Message = IndexGenotypesMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, message))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        // Index the genotypes with the provided ids, using the provided encoder id
        unimplemented!()
    }

    fn max_attempts(&self) -> i32 {
        5
    }

    fn try_at(
        &self,
        _: i32,
        attempted_at: chrono::DateTime<chrono::Utc>,
    ) -> chrono::DateTime<chrono::Utc> {
        attempted_at + Duration::from_secs(10)
    }
}

// ============================================================
// TrainEncoder
// ============================================================

/// Message triggering a training of an encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainEncoderMessage {}

impl Message for TrainEncoderMessage {
    const NAME: &str = "TrainEncoder";
}

/// Handler processing encoder training jobs
pub struct TrainEncoderHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for TrainEncoderHandler {
    type Message = TrainEncoderMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, message))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        // Train a new model
        // Needs some way to figure out on what data
        unimplemented!()
    }

    fn max_attempts(&self) -> i32 {
        5
    }

    fn try_at(
        &self,
        _: i32,
        attempted_at: chrono::DateTime<chrono::Utc>,
    ) -> chrono::DateTime<chrono::Utc> {
        attempted_at + Duration::from_secs(10)
    }
}

// ============================================================
// Registration
// ============================================================

/// Registers all indexing job handlers with the job registry.
#[instrument(level = "debug", skip_all)]
pub fn register_job_handlers(
    service: &Arc<super::Service>,
    builder: fx_mq_jobs::RegistryBuilder,
) -> fx_mq_jobs::RegistryBuilder {
    builder
        .with_handler(TrainEncoderHandler {
            service: service.clone(),
        })
        .with_handler(IndexGenotypesHandler {
            service: service.clone(),
        })
}
