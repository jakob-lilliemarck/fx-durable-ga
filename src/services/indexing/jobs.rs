use super::repositories::encoders::Digest;
use fx_mq_jobs::{Message, Queries};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tracing::instrument;

/// Job to train an encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TrainEncoderMessage {
    indexer_id: Digest,
}

impl TrainEncoderMessage {
    pub fn new(indexer_id: Digest) -> Self {
        Self { indexer_id }
    }
}

impl Message for TrainEncoderMessage {
    const NAME: &str = "TrainEncoder";
}

/// Handler for encoder training jobs
pub(super) struct TrainEncoderHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for TrainEncoderHandler {
    type Message = TrainEncoderMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, _lease_renewer))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            let encoder_id = self.service.train_encoder(&message.indexer_id).await?;

            // Explicitly enables the encoder and informs the system through an event
            self.service.enable_encoder(encoder_id).await?;

            Ok(())
        })
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

/// Registers all indexing job handlers with the job registry.
#[instrument(level = "debug", skip_all)]
pub(super) fn register_job_handlers(
    builder: fx_mq_jobs::RegistryBuilder,
    service: &Arc<super::Service>,
    _: &Arc<Queries>,
) -> fx_mq_jobs::RegistryBuilder {
    builder.with_handler(TrainEncoderHandler {
        service: service.clone(),
    })
}
