use super::Error;
use crate::models::GeneBounds;
use crate::models::Morphology;
use chrono::{DateTime, Utc};
use tracing::instrument;

/// Database representation of a morphology with JSON-serialized gene bounds.
#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub(super) struct DBMorphology {
    pub(super) type_name: String,
    pub(super) type_hash: i32,
    pub(super) revised_at: DateTime<Utc>,
    pub(super) gene_bounds: serde_json::Value,
}

impl TryFrom<DBMorphology> for Morphology {
    type Error = Error;

    /// Converts database morphology to domain model by deserializing gene bounds.
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

    /// Converts domain morphology to database model by serializing gene bounds to JSON.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GeneBounds;
    use serde_json::json;

    fn create_test_morphology() -> Morphology {
        let gene_bounds = vec![
            GeneBounds::integer(0, 10, 11).unwrap(),
            GeneBounds::decimal(0.0, 1.0, 100, 2).unwrap(),
        ];
        Morphology::new("TestMorphology", 123, gene_bounds)
    }

    fn create_test_db_morphology() -> DBMorphology {
        let original_morphology = create_test_morphology();
        let mut db_morphology = DBMorphology::try_from(original_morphology).unwrap();

        // Create invalid gene bounds for error testing
        db_morphology.gene_bounds = json!({"invalid": "gene_bounds"});
        db_morphology
    }

    #[test]
    fn test_morphology_to_db_morphology_conversion() {
        let morphology = create_test_morphology();
        let db_morphology = DBMorphology::try_from(morphology.clone()).unwrap();

        assert_eq!(db_morphology.type_name, morphology.type_name);
        assert_eq!(db_morphology.type_hash, morphology.type_hash);
        assert_eq!(db_morphology.revised_at, morphology.revised_at);
        assert!(db_morphology.gene_bounds.is_array());
    }

    #[test]
    fn test_db_morphology_to_morphology_conversion() {
        // Create a real morphology and serialize it to get valid JSON
        let original_morphology = create_test_morphology();
        let db_morphology = DBMorphology::try_from(original_morphology).unwrap();

        // Now convert back
        let morphology = Morphology::try_from(db_morphology).unwrap();

        assert_eq!(morphology.type_name, "TestMorphology");
        assert_eq!(morphology.type_hash, 123);
        assert_eq!(morphology.gene_bounds.len(), 2);
    }

    #[test]
    fn test_invalid_gene_bounds_json_fails() {
        let db_morphology = create_test_db_morphology();

        let result = Morphology::try_from(db_morphology);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::Serde(_)));
    }
}
