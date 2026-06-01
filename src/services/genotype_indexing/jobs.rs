use crate::services::indexing::Digest;
use fx_mq_jobs::Queries;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tracing::instrument;
use uuid::Uuid;

/// Requests indexing of the specified genotypes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct IndexGenotypesMessage {
    pub genotype_ids: Vec<Uuid>,
    pub indexer_id: Digest,
    pub request_id: Option<Uuid>,
}

impl IndexGenotypesMessage {
    pub fn new(indexer_id: Digest, genotype_id: Vec<Uuid>, request_id: Option<Uuid>) -> Self {
        Self {
            genotype_ids: genotype_id,
            indexer_id,
            request_id,
        }
    }
}

impl fx_mq_jobs::Message for IndexGenotypesMessage {
    const NAME: &str = "IndexGenotypes";
}

/// Handles indexing of genotypes.
pub(super) struct IndexGenotypesHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for IndexGenotypesHandler {
    type Message = IndexGenotypesMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, _lease_renewer))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            let result = self
                .service
                .index_genotypes(
                    &message.indexer_id,
                    &message.genotype_ids,
                    message.request_id,
                )
                .await
                .map(|_| ());

            if let Err(error) = &result {
                tracing::error!(
                    message = "Error encountered processing IndexGenotype",
                    error = error.to_string()
                );
            }

            result
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

/// Requests retry of deferred indexation for an indexer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RetryDeferredIndexationMessage {
    pub indexer_id: Digest,
}

impl RetryDeferredIndexationMessage {
    pub fn new(indexer_id: Digest) -> Self {
        Self { indexer_id }
    }
}

impl fx_mq_jobs::Message for RetryDeferredIndexationMessage {
    const NAME: &str = "RetryDeferredIndexation";
}

/// Handles retry of deferred indexation.
pub(super) struct RetryDeferredIndexationHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for RetryDeferredIndexationHandler {
    type Message = RetryDeferredIndexationMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, _lease_renewer))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            let result = self
                .service
                .retry_deferred_indexation(&message.indexer_id)
                .await
                .map(|_| ());

            if let Err(error) = &result {
                tracing::error!(
                    message = "Error encountered processing RetryDeferredIndexation",
                    error = error.to_string()
                );
            }

            result
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
    builder
        .with_handler(IndexGenotypesHandler {
            service: service.clone(),
        })
        .with_handler(RetryDeferredIndexationHandler {
            service: service.clone(),
        })
}
