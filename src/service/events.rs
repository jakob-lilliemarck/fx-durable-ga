use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Serialize, Deserialize)]
pub struct OptimizationRequested {
    request_id: Uuid,
}

impl fx_event_bus::Event for OptimizationRequested {
    const NAME: &'static str = "OptimizationRequested";
}

impl OptimizationRequested {
    pub fn new(request_id: Uuid) -> Self {
        Self { request_id }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GenotypeGenerated {
    request_id: Uuid,
    genotype_id: Uuid,
}

impl fx_event_bus::Event for GenotypeGenerated {
    const NAME: &'static str = "GenotypeGenerated";
}

impl GenotypeGenerated {
    pub fn new(request_id: Uuid, genotype_id: Uuid) -> Self {
        Self {
            request_id,
            genotype_id,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GenotypeEvaluated {
    request_id: Uuid,
    genotype_id: Uuid,
}

impl fx_event_bus::Event for GenotypeEvaluated {
    const NAME: &'static str = "GenotypeEvaluated";
}

impl GenotypeEvaluated {
    pub fn new(request_id: Uuid, genotype_id: Uuid) -> Self {
        Self {
            request_id,
            genotype_id,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RequestCompleted {
    request_id: Uuid,
}

impl fx_event_bus::Event for RequestCompleted {
    const NAME: &'static str = "RequestCompleted";
}

impl RequestCompleted {
    pub fn new(request_id: Uuid) -> Self {
        Self { request_id }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RequestTerminated {
    request_id: Uuid,
}

impl fx_event_bus::Event for RequestTerminated {
    const NAME: &'static str = "RequestTerminated";
}

impl RequestTerminated {
    pub fn new(request_id: Uuid) -> Self {
        Self { request_id }
    }
}
