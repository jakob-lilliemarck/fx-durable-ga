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
