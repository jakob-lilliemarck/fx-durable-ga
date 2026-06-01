use fx_mq_jobs::{Handler, Message, Queries};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tracing::instrument;
use uuid::Uuid;

// ============================================================
// Add optimization budget
// ============================================================
/// Adds a budget to an optimization account.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AddOptimizationBudgetMessage {
    pub account_id: Uuid,
    pub amount: i64,
    pub reason: String,
}

impl AddOptimizationBudgetMessage {
    pub fn new(account_id: Uuid, amount: i64, reason: String) -> Self {
        Self {
            account_id,
            amount,
            reason,
        }
    }
}

impl Message for AddOptimizationBudgetMessage {
    const NAME: &str = "AddOptimizationBudget";
}

/// Adds budget to an optimization account.
pub(super) struct AddOptimizationBudgetHandler {
    service: Arc<super::Service>,
}

impl Handler for AddOptimizationBudgetHandler {
    type Message = AddOptimizationBudgetMessage;
    type Error = super::Error;

    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            self.service
                .budgeting
                .append(
                    message.account_id,
                    super::service::ACCOUNT_TYPE.to_string(),
                    message.amount,
                    message.reason,
                )
                .await?;

            Ok(())
        })
    }

    fn max_attempts(&self) -> i32 {
        5
    }

    fn try_at(
        &self,
        _attempted: i32,
        attempted_at: chrono::DateTime<chrono::Utc>,
    ) -> chrono::DateTime<chrono::Utc> {
        attempted_at + Duration::from_secs(10)
    }
}

// ============================================================
// Charge optimization budget
// ============================================================
/// Charges the optimization budget for a batch of evaluations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ChargeOptimizationBudgetMessage {
    account_id: Uuid,
    amount: i64,
    last_charge_id: Option<Uuid>,
}

impl ChargeOptimizationBudgetMessage {
    pub fn new(account_id: Uuid, amount: i64, last_charge_id: Option<Uuid>) -> Self {
        Self {
            account_id,
            amount,
            last_charge_id,
        }
    }
}

impl Message for ChargeOptimizationBudgetMessage {
    const NAME: &str = "ChargeOptimizationBudget";
}

/// Handles charging the optimization budget.
pub(super) struct ChargeOptimizationBudgetHandler {
    service: Arc<super::Service>,
}

impl Handler for ChargeOptimizationBudgetHandler {
    type Message = ChargeOptimizationBudgetMessage;
    type Error = super::Error;

    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            self.service
                .try_charge(message.account_id, message.amount, message.last_charge_id)
                .await?;
            Ok(())
        })
    }

    fn max_attempts(&self) -> i32 {
        5
    }

    fn try_at(
        &self,
        _attempted: i32,
        attempted_at: chrono::DateTime<chrono::Utc>,
    ) -> chrono::DateTime<chrono::Utc> {
        attempted_at + Duration::from_secs(10)
    }
}

// ============================================================
// Breed genotypes
//
// If the next generation is the first generation, this job
// will seed the initial population, otherwise breed from last
// generation.
// ============================================================
/// Requests breeding of the next generation of genotypes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct BreedGenotypesMessage {
    request_id: Uuid,
    count: i64,
}

impl BreedGenotypesMessage {
    pub(crate) fn new(request_id: Uuid, count: i64) -> Self {
        Self { request_id, count }
    }
}

impl Message for BreedGenotypesMessage {
    const NAME: &str = "BreedGenotypes";
}

/// Handler that processes genotype breeding jobs.
pub(super) struct BreedGenotypesHandler {
    service: Arc<super::Service>,
}

impl Handler for BreedGenotypesHandler {
    type Message = BreedGenotypesMessage;
    type Error = super::Error;

    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            self.service
                .breed_genotypes(message.request_id, message.count)
                .await?;
            Ok(())
        })
    }

    fn max_attempts(&self) -> i32 {
        5
    }

    fn try_at(
        &self,
        _attempted: i32,
        attempted_at: chrono::DateTime<chrono::Utc>,
    ) -> chrono::DateTime<chrono::Utc> {
        attempted_at + Duration::from_secs(10)
    }
}
// ============================================================
// Evaluate a genotype
// ============================================================
/// Message requesting evaluation of a specific genotype.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluateGenotypeMessage {
    pub request_id: Uuid,
    pub genotype_id: Uuid,
}

impl EvaluateGenotypeMessage {
    pub(crate) fn new(request_id: Uuid, genotype_id: Uuid) -> Self {
        Self {
            request_id,
            genotype_id,
        }
    }
}

impl Message for EvaluateGenotypeMessage {
    const NAME: &str = "EvaluateGenotype";
}

/// Handles evaluating a single genotype.
pub(super) struct EvaluateGenotypeHandler {
    service: Arc<super::Service>,
}

impl Handler for EvaluateGenotypeHandler {
    type Message = EvaluateGenotypeMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, _lease_renewer))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            self.service
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

/// Triggers population maintenance after a genotype evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MaintainPopulationMessage {
    request_id: Uuid,
    genotype_id: Uuid,
    fitness: f64,
}

impl MaintainPopulationMessage {
    pub fn new(request_id: Uuid, genotype_id: Uuid, fitness: f64) -> Self {
        Self {
            request_id,
            genotype_id,
            fitness,
        }
    }
}

impl Message for MaintainPopulationMessage {
    const NAME: &str = "MaintainPopulation";
}

/// Handles population maintenance after each evaluation.
pub(super) struct MaintainPopulationHandler {
    service: Arc<super::Service>,
}

impl Handler for MaintainPopulationHandler {
    type Message = MaintainPopulationMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, _lease_renewer))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        let service = self.service.clone();
        Box::pin(async move {
            service
                .should_breed_next_generation(message.request_id, message.fitness)
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

/// Registers all optimization job handlers with the job registry.
#[instrument(level = "debug", skip_all)]
pub(super) fn register_job_handlers(
    builder: fx_mq_jobs::RegistryBuilder,
    service: &Arc<super::Service>,
    _: &Arc<Queries>,
) -> fx_mq_jobs::RegistryBuilder {
    builder
        .with_handler(AddOptimizationBudgetHandler {
            service: service.clone(),
        })
        .with_handler(ChargeOptimizationBudgetHandler {
            service: service.clone(),
        })
        .with_handler(BreedGenotypesHandler {
            service: service.clone(),
        })
        .with_handler(MaintainPopulationHandler {
            service: service.clone(),
        })
        .with_handler(EvaluateGenotypeHandler {
            service: service.clone(),
        })
}
