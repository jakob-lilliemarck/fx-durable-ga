use super::Error;
use crate::models::GeneBounds;
use crate::models::Morphology;
use chrono::{DateTime, Utc};
use tracing::instrument;

#[derive(Debug)]
pub(super) struct DBMorphology {
    pub(super) type_name: String,
    pub(super) type_hash: i32,
    pub(super) revised_at: DateTime<Utc>,
    pub(super) gene_bounds: serde_json::Value,
}

impl TryFrom<DBMorphology> for Morphology {
    type Error = Error;

    #[instrument(level = "debug", fields(type_name = %db_morphology.type_name, type_hash = db_morphology.type_hash))]
    fn try_from(db_morphology: DBMorphology) -> Result<Self, Self::Error> {
        let gene_bounds: Vec<GeneBounds> = serde_json::from_value(db_morphology.gene_bounds)?;

        Ok(Morphology {
            type_hash: db_morphology.type_hash,
            type_name: db_morphology.type_name,
            revised_at: db_morphology.revised_at,
            gene_bounds: gene_bounds,
        })
    }
}

impl TryFrom<Morphology> for DBMorphology {
    type Error = Error;

    #[instrument(level = "debug", fields(type_name = %morphology.type_name, type_hash = morphology.type_hash))]
    fn try_from(morphology: Morphology) -> Result<Self, Self::Error> {
        let gene_bounds_json = serde_json::to_value(&morphology.gene_bounds)?;

        Ok(DBMorphology {
            type_hash: morphology.type_hash,
            type_name: morphology.type_name,
            revised_at: morphology.revised_at,
            gene_bounds: gene_bounds_json,
        })
    }
}
