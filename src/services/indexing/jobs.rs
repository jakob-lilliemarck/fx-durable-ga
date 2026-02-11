use fx_mq_jobs::Message;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tracing::instrument;

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

    #[instrument(level = "debug", skip(self))]
    fn handle<'a>(
        &'a self,
        _: Self::Message,
        _: fx_mq_jobs::LeaseRenewer,
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
    builder.with_handler(TrainEncoderHandler {
        service: service.clone(),
    })
}
