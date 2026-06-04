use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenotypeEvaluatedEvent {
    pub evaluation_id: Uuid,
    pub request_id: Option<Uuid>,
    pub genotype_id: Uuid,
    pub fitness: f64,
}

impl fx_event_bus::Event for GenotypeEvaluatedEvent {
    const NAME: &'static str = "GenotypeEvaluated";
}

impl GenotypeEvaluatedEvent {
    pub fn new(
        evaluation_id: Uuid,
        request_id: Option<Uuid>,
        genotype_id: Uuid,
        fitness: f64,
    ) -> Self {
        Self {
            evaluation_id,
            request_id,
            genotype_id,
            fitness,
        }
    }
}
