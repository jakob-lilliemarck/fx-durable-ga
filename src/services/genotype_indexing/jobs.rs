use crate::{GenotypesFilter, services::genotype_indexing::service::GroupingKey};
use fx_mq_jobs::Queries;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tracing::instrument;
use uuid::Uuid;

// ============================================================
// IndexGenotypes
//
// Indexes any number or selection of genotypes by dispatching
// jobs scoped at more specific tagging groups
// ============================================================

/// Message to trigger indexing batch job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupGenotypesMessage {
    pub request_ids: Option<Vec<Uuid>>,
    pub generation_ids: Option<Vec<i32>>,
    pub genotype_ids: Option<Vec<Uuid>>,
}

impl fx_mq_jobs::Message for GroupGenotypesMessage {
    const NAME: &str = "GroupGenotypes";
}

/// Handler that processes requests for indexation of any number
/// or selection of genotypes, chunking the request to standard
/// tag groups.
pub struct GroupGenotypesHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for GroupGenotypesHandler {
    type Message = GroupGenotypesMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, message))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        let mut filter = GenotypesFilter::default();

        if let Some(request_ids) = message.request_ids {
            for id in request_ids {
                filter = filter.with_request_id(id);
            }
        }

        if let Some(generation_ids) = message.generation_ids {
            for id in generation_ids {
                filter = filter.with_generation_id(id);
            }
        }

        if let Some(genotype_ids) = message.genotype_ids {
            for id in genotype_ids {
                filter = filter.with_genotype_id(id);
            }
        }

        Box::pin(async move {
            self.service.group_genotypes(&filter).await?;
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
// IndexGenotypeGroup - index a group of genotypes
// ============================================================

/// Message to trigger indexing batch job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexGroupMessage {
    pub grouping_key: GroupingKey,
    pub genotype_ids: Vec<Uuid>,
}

impl fx_mq_jobs::Message for IndexGroupMessage {
    const NAME: &str = "IndexGroup";
}

/// Handler that processes indexing batch-jobs
pub struct IndexGroupHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for IndexGroupHandler {
    type Message = IndexGroupMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, message))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            match self
                .service
                .index_genotypes(&message.grouping_key, &message.genotype_ids)
                .await
            {
                Err(err) => {
                    tracing::error!(
                        message = "Failed to process IndexGenotypes",
                        error = err.to_string()
                    );
                    Err(err)
                }
                Ok(_) => Ok(()),
            }
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
        .with_handler(GroupGenotypesHandler {
            service: service.clone(),
        })
        .with_handler(IndexGroupHandler {
            service: service.clone(),
        })
}
