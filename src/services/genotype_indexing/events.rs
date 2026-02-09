use super::jobs::GroupGenotypesMessage;
use crate::services::{
    genotype_indexing::{jobs::IndexGroupMessage, service::GroupingKey},
    optimization::GenotypeEvaluatedEvent,
};
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
                .publish(&GroupGenotypesMessage {
                    request_ids: Some(vec![input.request_id]),
                    generation_ids: Some(vec![input.generation_id]),
                    genotype_ids: Some(vec![input.genotype_id]),
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
// IndexingGroupCreatedEvent
// ============================================================

#[derive(Clone, Serialize, Deserialize)]
pub struct GroupCreatedEvent {
    pub(super) grouping_key: GroupingKey,
    pub(super) genotype_ids: Vec<Uuid>,
}

impl fx_event_bus::Event for GroupCreatedEvent {
    const NAME: &'static str = "GroupCreated";
}

pub struct GroupCreatedHandler {
    queries: Arc<Queries>,
}

impl Handler<GroupCreatedEvent> for GroupCreatedHandler {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "debug", skip(self, input, tx))]
    fn handle<'a>(
        &'a self,
        input: Arc<GroupCreatedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            let mut publisher = fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx, &self.queries);

            let result = match publisher
                .publish(&IndexGroupMessage {
                    grouping_key: input.grouping_key.clone(),
                    genotype_ids: input.genotype_ids.clone(),
                })
                .await
            {
                Err(err) => {
                    tracing::error!(
                        message = "Failed to publish IndexGenotypes job",
                        grouping_key = ?input.grouping_key,
                        genotype_ids = ?input.genotype_ids

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
    registry.with_handler(GroupCreatedHandler {
        queries: queries.clone(),
    });
}
