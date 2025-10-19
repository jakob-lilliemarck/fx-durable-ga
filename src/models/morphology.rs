use crate::models::{Gene, GeneBounds};
use chrono::{DateTime, Utc};
use tracing::instrument;

#[derive(Debug)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct Morphology {
    pub(crate) revised_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) gene_bounds: Vec<GeneBounds>,
}

impl Morphology {
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash, gene_bounds_count = gene_bounds.len()))]
    pub(crate) fn new(type_name: &str, type_hash: i32, gene_bounds: Vec<GeneBounds>) -> Self {
        Self {
            revised_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            gene_bounds,
        }
    }

    #[instrument(level = "debug", fields(type_name = %self.type_name, type_hash = self.type_hash, gene_bounds_count = self.gene_bounds.len()))]
    pub(crate) fn random(&self) -> Vec<Gene> {
        let mut rng = rand::rng();

        self.gene_bounds
            .iter()
            .map(|gene_bound| gene_bound.random(&mut rng))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::SubsecRound;

    fn create_test_gene_bounds() -> Vec<GeneBounds> {
        vec![
            GeneBounds::integer(0, 10, 11).unwrap(),
            GeneBounds::integer(-5, 5, 11).unwrap(),
            GeneBounds::decimal(0.0, 1.0, 100, 2).unwrap(),
        ]
    }

    #[test]
    fn test_morphology_new() {
        let gene_bounds = create_test_gene_bounds();
        let before_creation = Utc::now();

        let morphology = Morphology::new("TestMorphology", 42, gene_bounds.clone());

        let after_creation = Utc::now();

        // Verify all fields are set correctly
        assert_eq!(morphology.type_name, "TestMorphology");
        assert_eq!(morphology.type_hash, 42);
        assert_eq!(morphology.gene_bounds, gene_bounds);

        // Verify timestamp is reasonable (within the test execution window)
        assert!(morphology.revised_at >= before_creation.trunc_subsecs(6));
        assert!(morphology.revised_at <= after_creation);
    }

    #[test]
    fn test_morphology_new_empty_gene_bounds() {
        let morphology = Morphology::new("EmptyMorphology", 0, vec![]);

        assert_eq!(morphology.type_name, "EmptyMorphology");
        assert_eq!(morphology.type_hash, 0);
        assert!(morphology.gene_bounds.is_empty());
    }

    #[test]
    fn test_random_generates_correct_genome_size() {
        let gene_bounds = create_test_gene_bounds();
        let morphology = Morphology::new("TestMorphology", 42, gene_bounds);

        let genome = morphology.random();

        // Should generate exactly as many genes as bounds
        assert_eq!(genome.len(), 3);
    }

    #[test]
    fn test_random_respects_gene_bounds() {
        let gene_bounds = vec![
            GeneBounds::integer(10, 20, 11).unwrap(), // 11 steps (0-10)
            GeneBounds::integer(-10, -5, 6).unwrap(), // 6 steps (0-5)
            GeneBounds::decimal(0.5, 1.5, 100, 2).unwrap(), // 100 steps (0-99)
        ];
        let morphology = Morphology::new("BoundedMorphology", 1, gene_bounds);

        // Test multiple generations to ensure consistency
        for _ in 0..10 {
            let genome = morphology.random();

            // First gene: 11 steps, so gene indices 0-10
            assert!((0..11).contains(&genome[0]));

            // Second gene: 6 steps, so gene indices 0-5
            assert!((0..6).contains(&genome[1]));

            // Third gene: 100 steps, so gene indices 0-99
            assert!((0..100).contains(&genome[2]));
        }
    }

    #[test]
    fn test_random_produces_different_genomes() {
        let gene_bounds = vec![
            GeneBounds::integer(0, 1000, 1001).unwrap(),
            GeneBounds::integer(0, 1000, 1001).unwrap(),
            GeneBounds::integer(0, 1000, 1001).unwrap(),
        ];
        let morphology = Morphology::new("RandomMorphology", 1, gene_bounds);

        let genome1 = morphology.random();
        let genome2 = morphology.random();

        // With large ranges, genomes should be different
        // (extremely unlikely to be identical)
        assert_ne!(genome1, genome2);
    }

    #[test]
    fn test_random_with_single_gene_bound() {
        let gene_bounds = vec![GeneBounds::integer(5, 15, 11).unwrap()]; // 11 steps (0-10)
        let morphology = Morphology::new("SingleGeneMorphology", 1, gene_bounds);

        let genome = morphology.random();

        assert_eq!(genome.len(), 1);
        assert!((0..11).contains(&genome[0])); // Gene indices are 0-10
    }

    #[test]
    fn test_random_with_empty_gene_bounds() {
        let morphology = Morphology::new("EmptyMorphology", 1, vec![]);

        let genome = morphology.random();

        assert!(genome.is_empty());
    }

    #[test]
    fn test_morphology_clone_and_equality() {
        let gene_bounds = create_test_gene_bounds();
        let morphology1 = Morphology::new("TestMorphology", 42, gene_bounds.clone());
        let mut morphology2 = morphology1.clone();

        // Initially equal
        assert_eq!(morphology1, morphology2);

        // Modify one field to test inequality
        morphology2.type_hash = 43;
        assert_ne!(morphology1, morphology2);
    }
}
