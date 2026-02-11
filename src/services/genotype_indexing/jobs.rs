use crate::services::genotype_indexing::service::GroupingKey;
use fx_mq_jobs::Queries;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tracing::instrument;
use uuid::Uuid;

// ============================================================
// IndexGenotypes
//
// Indexes a group of genotypes that share the same GroupingKey
// ============================================================

/// Message to trigger indexing batch job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexGenotypesMessage {
    pub grouping_key: GroupingKey,
    pub genotype_ids: Vec<Uuid>,
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
        Box::pin(async move {
            if let Err(error) = self
                .service
                .index_genotype_group(&message.grouping_key, &message.genotype_ids)
                .await
            {
                tracing::error!(
                    message = "Error encountered processing IndexGenotypeGroup",
                    error = error.to_string()
                );
                return Err(error);
            }
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

// ============================================================
// IndexGenotype
//
// Indexes a single Genotype
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexGenotypeMessage {
    pub genotype_id: Uuid,
}

impl fx_mq_jobs::Message for IndexGenotypeMessage {
    const NAME: &str = "IndexGenotype";
}

pub struct IndexGenotypeHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for IndexGenotypeHandler {
    type Message = IndexGenotypeMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, message))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            if let Err(error) = self.service.index_genotype(&message.genotype_id).await {
                tracing::error!(
                    message = "Error encountered processing IndexGenotype",
                    error = error.to_string()
                );
                return Err(error);
            }
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

// ============================================================
// Registration
// ============================================================

/// Registers all indexing job handlers with the job registry.
#[instrument(level = "debug", skip_all)]
pub fn register_job_handlers(
    service: &Arc<super::Service>,
    builder: fx_mq_jobs::RegistryBuilder,
    _: Arc<Queries>,
) -> fx_mq_jobs::RegistryBuilder {
    builder
        .with_handler(IndexGenotypesHandler {
            service: service.clone(),
        })
        .with_handler(IndexGenotypeHandler {
            service: service.clone(),
        })
}
