use crate::services::{
    genotype_indexing::jobs::IndexGenotypeMessage, optimization::GenotypeEvaluatedEvent,
};
use fx_event_bus::Handler;
use fx_mq_jobs::Queries;
use sqlx::PgTransaction;
use std::sync::Arc;
use tracing::instrument;

// ============================================================
// GenotypeEvaluated
// ============================================================

pub struct GenotypeEvaluatedHandler {
    queries: Arc<Queries>,
}

impl Handler<GenotypeEvaluatedEvent> for GenotypeEvaluatedHandler {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "debug", skip(self, input, tx))]
    fn handle<'a>(
        &'a self,
        input: Arc<GenotypeEvaluatedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            let mut publisher = fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx, &self.queries);

            let result = publisher
                .publish(&IndexGenotypeMessage {
                    genotype_id: input.genotype_id,
                })
                .await
                .map(|_| ());

            if let Err(error) = result {
                tracing::error!(
                    message = "Failed to publish IndexGenotype",
                    genotype_id = %input.genotype_id,
                    error=error.to_string()
                );
                return (publisher.into(), Err(error));
            }

            (publisher.into(), result)
        })
    }
}

// ============================================================
// Registration
// ============================================================

/// Registers all optimization event handlers with the event bus registry.
#[instrument(level = "debug", skip_all)]
pub fn register_event_handlers(
    queries: Arc<Queries>,
    registry: &mut fx_event_bus::EventHandlerRegistry,
) {
    registry.with_handler(GenotypeEvaluatedHandler {
        queries: queries.clone(),
    });
}
