use crate::services::{
    genotype_indexing::jobs::RetryDeferredIndexationMessage,
    indexing::EncoderAvailableEvent,
};
use fx_event_bus::Handler;
use fx_mq_jobs::Queries;
use sqlx::PgTransaction;
use std::sync::Arc;
use tracing::instrument;

/// Handles encoder availability changes by retrying deferred indexation.
pub struct EncoderAvailabilityChangedHandler {
    queries: Arc<Queries>,
}

impl Handler<EncoderAvailableEvent> for EncoderAvailabilityChangedHandler {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "debug", skip(self, tx))]
    fn handle<'a>(
        &'a self,
        event: Arc<EncoderAvailableEvent>,
        _polled_at: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            let mut publisher = fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx, &self.queries);
            let result = publisher
                .publish(&RetryDeferredIndexationMessage::new(event.indexer_id))
                .await
                .map(|_| ());

            (publisher.into(), result)
        })
    }
}

/// Registers all optimization event handlers with the event bus registry.
#[instrument(level = "debug", skip_all)]
pub(super) fn register_event_handlers(
    registry: &mut fx_event_bus::EventHandlerRegistry,
    queries: &Arc<Queries>,
) {
    registry.with_handler(EncoderAvailabilityChangedHandler {
        queries: queries.clone(),
    });
}
