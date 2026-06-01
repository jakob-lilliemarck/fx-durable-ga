use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const SEMAPHORE_CHANNEL: &str = "fx_durable_ga_semaphores";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Semaphore {
    pub(super) name: String,
    pub(super) raised_at: DateTime<Utc>,
}

impl Semaphore {
    pub fn raised_at(&self) -> &DateTime<Utc> {
        &self.raised_at
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}
