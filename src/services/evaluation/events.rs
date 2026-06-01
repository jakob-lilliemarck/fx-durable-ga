use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenotypeEvaluatedEvent {
    pub request_id: Option<Uuid>,
    pub genotype_id: Uuid,
    pub fitness: f64,
}

impl fx_event_bus::Event for GenotypeEvaluatedEvent {
    const NAME: &'static str = "GenotypeEvaluated";
}

impl GenotypeEvaluatedEvent {
    pub fn new(request_id: Option<Uuid>, genotype_id: Uuid, fitness: f64) -> Self {
        Self {
            request_id,
            genotype_id,
            fitness,
        }
    }
}
