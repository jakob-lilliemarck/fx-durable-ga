use fx_mq_building_blocks::models::Message;
use fx_mq_jobs::Handler;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tracing::instrument;
use uuid::Uuid;

// ============================================================
// GenerateInitialPopulation
// ============================================================

/// Message to trigger initial population generation for an optimization request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateInitialPopulationMessage {
    pub request_id: Uuid,
}

impl Message for GenerateInitialPopulationMessage {
    const NAME: &str = "GenerateInitialPopulationMessage";
}

/// Handler that processes initial population generation jobs.
pub struct GenerateInitialPopulationHandler {
    service: Arc<super::Service>,
}

impl Handler for GenerateInitialPopulationHandler {
    type Message = GenerateInitialPopulationMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, message), fields(request_id = %message.request_id))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        let service = self.service.clone();
        Box::pin(async move {
            service
                .generate_initial_population(message.request_id)
                .await
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
// EvaluateGenotype
// ============================================================

/// Message to trigger fitness evaluation for a specific genotype.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluateGenotypeMessage {
    pub request_id: Uuid,
    pub genotype_id: Uuid,
}

impl Message for EvaluateGenotypeMessage {
    const NAME: &str = "EvaluateGenotypeMessage";
}

/// Handler that processes genotype evaluation jobs.
pub struct EvaluateGenotypeHandler {
    service: Arc<super::Service>,
}

impl Handler for EvaluateGenotypeHandler {
    type Message = EvaluateGenotypeMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, message), fields(request_id = %message.request_id, genotype_id = %message.genotype_id))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        let service = self.service.clone();
        Box::pin(async move {
            service
                .evaluate_genotype(message.request_id, message.genotype_id)
                .await
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
// MaintainPopulation
// ============================================================

/// Message to trigger population maintenance for an optimization request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintainPopulationMessage {
    pub request_id: Uuid,
}

impl Message for MaintainPopulationMessage {
    const NAME: &str = "MaintainPopulationMessage";
}

/// Handler that processes population maintenance jobs.
pub struct MaintainPopulationHandler {
    service: Arc<super::Service>,
}

impl Handler for MaintainPopulationHandler {
    type Message = MaintainPopulationMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, message), fields(request_id = %message.request_id))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        let service = self.service.clone();
        Box::pin(async move { service.maintain_population(message.request_id).await })
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

/// Registers all optimization job handlers with the job registry.
#[instrument(level = "debug", skip_all)]
pub fn register_job_handlers(
    service: &Arc<super::Service>,
    builder: fx_mq_jobs::RegistryBuilder,
) -> fx_mq_jobs::RegistryBuilder {
    builder
        .with_handler(GenerateInitialPopulationHandler {
            service: service.clone(),
        })
        .with_handler(EvaluateGenotypeHandler {
            service: service.clone(),
        })
        .with_handler(MaintainPopulationHandler {
            service: service.clone(),
        })
}
