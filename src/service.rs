use crate::{gene::Population, registry::Registry};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("registry error: {0}")]
    Registry(#[from] crate::registry::RegistryError),
}

pub enum PopulationStatus {
    InProgress { remaining: u32 },
    Completed,
}

pub enum OptimizationStatus {
    TargetReached { target: f64, actual: f64 },
    IterationsExhausted,
}

pub struct GAOptimization {
    id: Uuid,
    registry: Arc<Registry>,
    population: Population,
}

// optimization service
pub struct Service {}

impl Service {
    pub fn new_optimization() -> Result<GAOptimization, ServiceError> {
        todo!()
    }

    pub fn next_population(&self) -> Result<Population, ServiceError> {
        todo!()
    }

    pub fn evaluate_phenotype(&self) -> Result<f64, ServiceError> {
        todo!()
    }

    pub fn check_population_status(&self) -> Result<PopulationStatus, ServiceError> {
        todo!()
    }

    pub fn check_optimization_status(&self) -> Result<OptimizationStatus, ServiceError> {
        todo!()
    }
}
