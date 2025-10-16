use sqlx::prelude::FromRow;
use uuid::Uuid;

#[derive(Debug, FromRow)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Individual {
    pub(crate) genotype_id: Uuid,
    pub(crate) request_id: Uuid,
    pub(crate) generation_id: i32,
}

impl Individual {
    pub fn new(genotype_id: Uuid, request_id: Uuid, generation_id: i32) -> Self {
        Self {
            genotype_id,
            request_id,
            generation_id,
        }
    }
}
