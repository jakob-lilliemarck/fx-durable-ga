use chrono::{DateTime, Utc};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, FromRow)]
pub struct NoiseProbe {
    pub(super) id: Uuid,
    pub(super) genotype_id: Uuid,
    pub(super) request_id: Uuid,
    pub(super) evaluation_count: i32,
    pub(super) created_at: DateTime<Utc>,
}

impl NoiseProbe {
    pub fn new(genotype_id: Uuid, request_id: Uuid, evaluation_count: i32) -> Self {
        Self {
            id: Uuid::now_v7(),
            genotype_id,
            request_id,
            evaluation_count,
            created_at: Utc::now(),
        }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn genotype_id(&self) -> Uuid {
        self.genotype_id
    }

    pub fn request_id(&self) -> Uuid {
        self.request_id
    }

    pub fn evaluation_count(&self) -> i32 {
        self.evaluation_count
    }
}

#[derive(Default, Debug)]
pub struct SearchNoiseProbesFilter {
    pub(super) request_id: Option<Uuid>,
}

impl SearchNoiseProbesFilter {
    pub fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_id = Some(request_id);
        self
    }
}
