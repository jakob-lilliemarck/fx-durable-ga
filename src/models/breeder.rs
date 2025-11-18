use crate::models::{Genotype, Morphology, Request};
use tracing::instrument;

/// Handles the breeding process by combining crossover and mutation operations.
pub(crate) struct Breeder;

impl Breeder {
    /// Creates a single child from two parents using crossover and mutation.
    #[instrument(level = "debug", skip(request, morphology, parent1, parent2, rng), fields(parent1_id = %parent1.id(), parent2_id = %parent2.id(), generation_id = next_generation_id, progress = progress))]
    fn breed_child(
        request: &Request,
        morphology: &Morphology,
        parent1: &Genotype,
        parent2: &Genotype,
        next_generation_id: i32,
        progress: f64,
        rng: &mut impl rand::Rng,
    ) -> Genotype {
        let genome = request.crossover.apply(rng, parent1, parent2);
        let mut child = Genotype::new(
            &request.type_name,
            request.type_hash,
            genome,
            request.id,
            next_generation_id,
        );

        // Apply mutation based on current optimization progress
        request
            .mutagen
            .mutate(rng, &mut child, morphology, progress);

        child
    }

    /// Creates multiple children from parent pairs using crossover and mutation.
    #[instrument(level = "debug", skip(request, morphology, parent_pairs, rng), fields(num_pairs = parent_pairs.len(), generation_id = next_generation_id, progress = progress))]
    pub(crate) fn breed_batch(
        request: &Request,
        morphology: &Morphology,
        parent_pairs: &[(&Genotype, &Genotype)],
        next_generation_id: i32,
        progress: f64,
        rng: &mut impl rand::Rng,
    ) -> Vec<Genotype> {
        parent_pairs
            .iter()
            .map(|&(p1, p2)| {
                Self::breed_child(
                    request,
                    morphology,
                    p1,
                    p2,
                    next_generation_id,
                    progress,
                    rng,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        Crossover, Distribution, FitnessGoal, GeneBounds, Mutagen, MutationRate, Schedule,
        Selector, Temperature,
    };
    use chrono::Utc;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use uuid::Uuid;

    fn create_test_genotype(id: &str, genome: Vec<i64>) -> Genotype {
        Genotype {
            id: Uuid::parse_str(id).unwrap(),
            generated_at: Utc::now(),
            type_name: "test".to_string(),
            type_hash: 123,
            genome: genome.clone(),
            genome_hash: Genotype::compute_genome_hash(&genome),
            request_id: Uuid::now_v7(),
            generation_id: 1,
        }
    }

    fn create_test_request() -> Request {
        Request::new(
            "TestType",
            123,
            FitnessGoal::maximize(0.9).unwrap(),
            Selector::tournament(5, 20).expect("is valid"),
            Schedule::generational(100, 10),
            Mutagen::new(
                Temperature::constant(0.5).unwrap(),
                MutationRate::constant(0.1).unwrap(),
            ),
            Crossover::uniform(0.5).unwrap(),
            Distribution::latin_hypercube(50),
            None::<()>,
        )
        .unwrap()
    }

    fn create_test_morphology() -> Morphology {
        Morphology::new(
            "TestType",
            123,
            vec![
                GeneBounds::integer(0, 10, 11).unwrap(), // 11 steps for range 0-10
                GeneBounds::integer(0, 10, 11).unwrap(),
                GeneBounds::integer(0, 10, 11).unwrap(),
            ],
        )
    }

    #[test]
    fn test_breed_batch_produces_correct_number_of_children() {
        let request = create_test_request();
        let morphology = create_test_morphology();
        let mut rng = StdRng::seed_from_u64(42);

        let parent1 = create_test_genotype("00000000-0000-0000-0000-000000000001", vec![1, 2, 3]);
        let parent2 = create_test_genotype("00000000-0000-0000-0000-000000000002", vec![4, 5, 6]);
        let parent3 = create_test_genotype("00000000-0000-0000-0000-000000000003", vec![7, 8, 9]);

        let parent_pairs = vec![(&parent1, &parent2), (&parent2, &parent3)];

        let children = Breeder::breed_batch(&request, &morphology, &parent_pairs, 2, 0.5, &mut rng);

        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_breed_batch_children_have_correct_metadata() {
        let request = create_test_request();
        let morphology = create_test_morphology();
        let mut rng = StdRng::seed_from_u64(42);

        let parent1 = create_test_genotype("00000000-0000-0000-0000-000000000001", vec![1, 2, 3]);
        let parent2 = create_test_genotype("00000000-0000-0000-0000-000000000002", vec![4, 5, 6]);

        let parent_pairs = vec![(&parent1, &parent2)];
        let next_generation_id = 5;

        let children = Breeder::breed_batch(
            &request,
            &morphology,
            &parent_pairs,
            next_generation_id,
            0.0,
            &mut rng,
        );

        let child = &children[0];
        assert_eq!(child.type_name(), "TestType");
        assert_eq!(child.type_hash(), 123);
        assert_eq!(child.request_id(), request.id);
        assert_eq!(child.generation_id(), next_generation_id);
        assert_eq!(child.genome().len(), 3); // Same length as parents
    }

    #[test]
    fn test_breed_batch_with_empty_parent_pairs() {
        let request = create_test_request();
        let morphology = create_test_morphology();
        let mut rng = StdRng::seed_from_u64(42);

        let parent_pairs: Vec<(&Genotype, &Genotype)> = vec![];

        let children = Breeder::breed_batch(&request, &morphology, &parent_pairs, 2, 0.5, &mut rng);

        assert_eq!(children.len(), 0);
    }

    #[test]
    fn test_breed_batch_children_are_unique() {
        let request = create_test_request();
        let morphology = create_test_morphology();
        let mut rng = StdRng::seed_from_u64(42);

        let parent1 = create_test_genotype("00000000-0000-0000-0000-000000000001", vec![1, 2, 3]);
        let parent2 = create_test_genotype("00000000-0000-0000-0000-000000000002", vec![4, 5, 6]);

        let parent_pairs = vec![(&parent1, &parent2), (&parent1, &parent2)];

        let children = Breeder::breed_batch(&request, &morphology, &parent_pairs, 2, 0.5, &mut rng);

        assert_eq!(children.len(), 2);

        // Each child should have a unique ID
        assert_ne!(children[0].id(), children[1].id());

        // Children should have valid (non-nil) UUIDs
        assert!(!children[0].id().is_nil());
        assert!(!children[1].id().is_nil());
    }

    #[test]
    fn test_breed_batch_respects_genome_bounds() {
        let request = create_test_request();
        let morphology = create_test_morphology(); // Bounds are 0-10 for each gene
        let mut rng = StdRng::seed_from_u64(42);

        let parent1 = create_test_genotype("00000000-0000-0000-0000-000000000001", vec![0, 5, 10]);
        let parent2 = create_test_genotype("00000000-0000-0000-0000-000000000002", vec![2, 7, 8]);

        let parent_pairs = vec![(&parent1, &parent2)];

        let children = Breeder::breed_batch(&request, &morphology, &parent_pairs, 2, 0.5, &mut rng);

        let child = &children[0];

        // All genes should be within bounds [0, 10]
        for &gene in &child.genome {
            assert!(gene >= 0);
            assert!(gene <= 10);
        }
    }

    #[test]
    fn test_breed_batch_children_differ_from_parents() {
        let request = create_test_request();
        let morphology = create_test_morphology();
        let mut rng = StdRng::seed_from_u64(123); // Different seed to ensure variance

        // Create parents with distinct genomes
        let parent1 = create_test_genotype("00000000-0000-0000-0000-000000000001", vec![1, 1, 1]);
        let parent2 = create_test_genotype("00000000-0000-0000-0000-000000000002", vec![9, 9, 9]);

        let parent_pairs = vec![(&parent1, &parent2)];

        let children = Breeder::breed_batch(
            &request,
            &morphology,
            &parent_pairs,
            2,
            0.8, // High progress value to ensure mutation occurs
            &mut rng,
        );

        let child = &children[0];

        // Child should be different from both parents due to crossover and mutation
        assert_ne!(child.genome, parent1.genome);
        assert_ne!(child.genome, parent2.genome);

        // Child should have same length as parents
        assert_eq!(child.genome.len(), parent1.genome.len());
        assert_eq!(child.genome.len(), parent2.genome.len());

        // Child's genome hash should differ from parents (confirms genetic diversity)
        assert_ne!(child.genome_hash, parent1.genome_hash);
        assert_ne!(child.genome_hash, parent2.genome_hash);
    }
}
