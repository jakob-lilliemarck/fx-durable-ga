use crate::models::evolution::GenotypeManager;
use crate::models::{Genotype, Request};
use tracing::instrument;

/// Handles the breeding process by combining crossover and mutation operations.
pub(crate) struct Breeder;

impl Breeder {
    /// Creates a single child from two parents using crossover and mutation.
    #[instrument(level = "debug", skip(request, manager, parent1, parent2, rng), fields(parent1_id = %parent1.id(), parent2_id = %parent2.id(), generation_id = next_generation_id, progress = progress, type_hash = request.type_hash))]
    fn breed_child(
        request: &Request,
        manager: &dyn GenotypeManager,
        parent1: &Genotype,
        parent2: &Genotype,
        next_generation_id: i32,
        progress: f64,
        rng: &mut dyn rand::RngCore,
    ) -> anyhow::Result<Genotype> {
        let p1 = parent1.genome().clone();
        let p2 = parent2.genome().clone();
        let mut child_genome = manager.crossover(&p1, &p2, rng, &request.user_defined)?;

        manager.mutate(&mut child_genome, rng, progress, &request.user_defined)?;

        let child = Genotype::new(
            &request.type_name,
            request.type_hash,
            child_genome,
            request.id,
            next_generation_id,
        );
        Ok(child)
    }

    /// Creates multiple children from parent pairs using crossover and mutation.
    #[instrument(level = "debug", skip(request, manager, parent_pairs, rng), fields(num_pairs = parent_pairs.len(), generation_id = next_generation_id, progress = progress, type_hash = request.type_hash))]
    pub(crate) fn breed_batch(
        request: &Request,
        manager: &dyn GenotypeManager,
        parent_pairs: &[(&Genotype, &Genotype)],
        next_generation_id: i32,
        progress: f64,
        rng: &mut dyn rand::RngCore,
    ) -> anyhow::Result<Vec<Genotype>> {
        let mut out = Vec::with_capacity(parent_pairs.len());
        for &(p1, p2) in parent_pairs {
            out.push(Self::breed_child(
                request,
                manager,
                p1,
                p2,
                next_generation_id,
                progress,
                rng,
            )?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::evolution::GenotypeManager;
    use crate::models::{FitnessGoal, Schedule, Selector, Terminated};
    use anyhow::Result;
    use futures::future::BoxFuture;
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use serde_json::Value;
    use uuid::Uuid;

    struct TestManager;
    impl GenotypeManager for TestManager {
        fn name(&self) -> &'static str {
            "test"
        }
        fn random(&self, rng: &mut dyn RngCore, _user: &Value) -> anyhow::Result<Value> {
            Ok(serde_json::json!([
                rng.random_range(0..10),
                rng.random_range(0..10)
            ]))
        }
        fn crossover(
            &self,
            parent1: &Value,
            parent2: &Value,
            _rng: &mut dyn RngCore,
            _user: &Value,
        ) -> Result<Value> {
            Ok(serde_json::json!([
                parent1[0].as_i64().unwrap_or(0),
                parent2[1].as_i64().unwrap_or(0)
            ]))
        }
        fn mutate(
            &self,
            genotype: &mut Value,
            _rng: &mut dyn RngCore,
            _progress: f64,
            _user: &Value,
        ) -> Result<()> {
            if let Some(first) = genotype.as_array_mut().and_then(|a| a.get_mut(0)) {
                *first = serde_json::json!(first.as_i64().unwrap_or(0) + 1);
            }
            Ok(())
        }
        fn evaluate<'a>(
            &'a self,
            _genotype: &'a Value,
            _terminated: &'a dyn Terminated,
            _user: &'a Value,
        ) -> BoxFuture<'a, Result<f64>> {
            Box::pin(async { Ok(1.0) })
        }
    }

    fn create_test_genotype(id: &str, genome: Value) -> Genotype {
        Genotype::new("test", 123, genome, Uuid::parse_str(id).unwrap(), 1)
    }

    fn create_test_request() -> Request {
        Request::new(
            "TestType",
            123,
            FitnessGoal::maximize(0.9).unwrap(),
            Selector::tournament(5, 20).expect("is valid"),
            Schedule::generational(100, 10),
            serde_json::json!({"probability":0.5}),
            None::<()>,
        )
        .unwrap()
    }

    #[test]
    fn test_breed_batch_produces_correct_number_of_children() {
        let request = create_test_request();
        let manager = TestManager;
        let mut rng = StdRng::seed_from_u64(42);

        let parent1 = create_test_genotype(
            "00000000-0000-0000-0000-000000000001",
            serde_json::json!([1, 2]),
        );
        let parent2 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([4, 5]),
        );
        let parent3 = create_test_genotype(
            "00000000-0000-0000-0000-000000000003",
            serde_json::json!([7, 8]),
        );

        let parent_pairs = vec![(&parent1, &parent2), (&parent2, &parent3)];
        let children =
            Breeder::breed_batch(&request, &manager, &parent_pairs, 2, 0.5, &mut rng).unwrap();

        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_breed_batch_children_have_correct_metadata() {
        let request = create_test_request();
        let manager = TestManager;
        let mut rng = StdRng::seed_from_u64(42);

        let parent1 = create_test_genotype(
            "00000000-0000-0000-0000-000000000001",
            serde_json::json!([1, 2]),
        );
        let parent2 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([4, 5]),
        );

        let parent_pairs = vec![(&parent1, &parent2)];
        let next_generation_id = 5;

        let children = Breeder::breed_batch(
            &request,
            &manager,
            &parent_pairs,
            next_generation_id,
            0.0,
            &mut rng,
        )
        .unwrap();

        let child = &children[0];
        assert_eq!(child.type_name(), "TestType");
        assert_eq!(child.type_hash(), 123);
        assert_eq!(child.request_id(), request.id);
        assert_eq!(child.generation_id(), next_generation_id);
    }

    #[test]
    fn test_breed_batch_with_empty_parent_pairs() {
        let request = create_test_request();
        let manager = TestManager;
        let mut rng = StdRng::seed_from_u64(42);

        let parent_pairs: Vec<(&Genotype, &Genotype)> = vec![];
        let children =
            Breeder::breed_batch(&request, &manager, &parent_pairs, 2, 0.5, &mut rng).unwrap();

        assert_eq!(children.len(), 0);
    }

    #[test]
    fn test_breed_batch_children_are_unique() {
        let request = create_test_request();
        let manager = TestManager;
        let mut rng = StdRng::seed_from_u64(42);

        let parent1 = create_test_genotype(
            "00000000-0000-0000-0000-000000000001",
            serde_json::json!([1, 2]),
        );
        let parent2 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([4, 5]),
        );

        let parent_pairs = vec![(&parent1, &parent2), (&parent1, &parent2)];
        let children =
            Breeder::breed_batch(&request, &manager, &parent_pairs, 2, 0.5, &mut rng).unwrap();

        assert_eq!(children.len(), 2);
        assert_ne!(children[0].id(), children[1].id());
        assert!(!children[0].id().is_nil());
        assert!(!children[1].id().is_nil());
    }
}
