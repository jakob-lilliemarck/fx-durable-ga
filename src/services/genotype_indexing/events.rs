use super::jobs::IndexGenotypesMessage;
use crate::services::optimization::GenotypeEvaluatedEvent;
use fx_event_bus::Handler;
use fx_mq_jobs::Queries;
use serde::{Deserialize, Serialize};
use sqlx::PgTransaction;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

// ============================================================
// GenotypeEvaluated - handler only
// ============================================================

/// Handler responding to completed genotype evaluations
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
        // NOTE!
        // For now just dispatches an indexing job for a single genotype
        // If database access becomes a bottleneck, then wait for N non-indexed
        // genotypes before dispatching.
        Box::pin(async move {
            let mut publisher = fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx, &self.queries);

            let result = match publisher
                .publish(&IndexGenotypesMessage {
                    genotype_ids: vec![input.genotype_id],
                })
                .await
            {
                Err(err) => {
                    tracing::error!(
                        message = "Failed to publish IndexGenotypes job",
                        genotype_id = %input.genotype_id
                    );
                    Err(err)
                }
                _ => Ok(()),
            };

            (publisher.into(), result)
        })
    }
}

// ============================================================
// GenotypeIndexed
// ============================================================

/// Event published when a genotype has been indexed
#[derive(Clone, Serialize, Deserialize)]
pub struct GenotypeIndexedEvent {
    genotype_id: Uuid,
}

impl fx_event_bus::Event for GenotypeIndexedEvent {
    const NAME: &'static str = "GenotypeIndexed";
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
