use crate::models::{Gene, Genotype};
use rand::Rng;
use serde::{Deserialize, Serialize};

fn crossover_uniform<R: Rng>(
    rng: &mut R,
    lhs: &Genotype,
    rhs: &Genotype,
    probability: f64,
) -> Vec<Gene> {
    lhs.genome
        .iter()
        .zip(rhs.genome.iter())
        .map(|(&lhs, &rhs)| {
            if rng.random_bool(probability) {
                lhs
            } else {
                rhs
            }
        })
        .collect()
}

fn crossover_single_point<R: Rng>(
    rng: &mut R,
    lhs: &Genotype,
    rhs: &Genotype,
    point: usize,
) -> Vec<Gene> {
    let mut genome = Vec::with_capacity(lhs.genome.len());

    genome.extend_from_slice(&lhs.genome[..point]); // First part from lhs
    genome.extend_from_slice(&rhs.genome[point..]); // Second part from rhs
    genome
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Crossover {
    Uniform { probability: f64 },
    SinglePoint,
}

impl Crossover {
    pub(crate) fn crossover<R: Rng>(
        &self,
        rng: &mut R,
        lhs: &Genotype,
        rhs: &Genotype,
    ) -> Vec<Gene> {
        match self {
            Self::Uniform { probability } => crossover_uniform(rng, lhs, rhs, *probability),
            Self::SinglePoint => {
                let point = rng.random_range(1..lhs.genome.len()); // Cut point
                crossover_single_point(rng, lhs, rhs, point)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};
    use uuid::Uuid;

    fn create_test_genotype(genome: Vec<Gene>) -> Genotype {
        Genotype::new("TestType", 123, genome, Uuid::now_v7(), 1)
    }

    #[test]
    fn it_performs_uniform_crossover() {
        let mut rng = StdRng::seed_from_u64(42);
        let parent_a = create_test_genotype(vec![1, 2, 3, 4, 5]);
        let parent_b = create_test_genotype(vec![6, 7, 8, 9, 10]);

        let child_genome = crossover_uniform(&mut rng, &parent_a, &parent_b, 0.5);

        // With seeded RNG, result should be deterministic
        assert_eq!(child_genome.len(), 5);

        // Each gene should be from either parent A or parent B
        for (i, &gene) in child_genome.iter().enumerate() {
            assert!(gene == parent_a.genome[i] || gene == parent_b.genome[i]);
        }
    }

    #[test]
    fn it_performs_single_point_crossover() {
        let mut rng = StdRng::seed_from_u64(42);
        let parent_a = create_test_genotype(vec![1, 2, 3, 4, 5]);
        let parent_b = create_test_genotype(vec![6, 7, 8, 9, 10]);

        let child_genome = crossover_single_point(&mut rng, &parent_a, &parent_b, 1);
        assert_eq!(child_genome, vec![1, 7, 8, 9, 10]);

        let child_genome = crossover_single_point(&mut rng, &parent_a, &parent_b, 2);
        assert_eq!(child_genome, vec![1, 2, 8, 9, 10]);

        let child_genome = crossover_single_point(&mut rng, &parent_a, &parent_b, 3);
        assert_eq!(child_genome, vec![1, 2, 3, 9, 10]);

        let child_genome = crossover_single_point(&mut rng, &parent_a, &parent_b, 4);
        assert_eq!(child_genome, vec![1, 2, 3, 4, 10]);
    }

    #[test]
    fn it_handles_single_point_crossover_via_enum() {
        let mut rng = StdRng::seed_from_u64(42);
        let parent_a = create_test_genotype(vec![1, 2, 3, 4, 5]);
        let parent_b = create_test_genotype(vec![6, 7, 8, 9, 10]);

        let crossover = Crossover::SinglePoint;
        let child = crossover.crossover(&mut rng, &parent_a, &parent_b);

        // Verify basic properties
        assert_eq!(child.len(), 5);

        // Verify there's exactly one transition point
        let mut transitions = 0;
        for i in 1..child.len() {
            let prev_from_a = child[i - 1] == parent_a.genome[i - 1];
            let curr_from_a = child[i] == parent_a.genome[i];

            if prev_from_a != curr_from_a {
                transitions += 1;
            }
        }

        // Single point crossover should have exactly 1 transition
        assert_eq!(
            transitions, 1,
            "Single point crossover should have exactly one transition point"
        );

        // All genes should come from one parent or the other
        for (i, &gene) in child.iter().enumerate() {
            assert!(gene == parent_a.genome[i] || gene == parent_b.genome[i]);
        }
    }

    #[test]
    fn it_handles_uniform_crossover_extreme_probabilities() {
        let mut rng = StdRng::seed_from_u64(42);

        let parent_a = create_test_genotype(vec![1, 2, 3]);
        let parent_b = create_test_genotype(vec![4, 5, 6]);

        // Probability 0.0 should always choose parent B
        let crossover = Crossover::Uniform { probability: 0.0 };
        let child_genome = crossover.crossover(&mut rng, &parent_a, &parent_b);
        assert_eq!(child_genome, parent_b.genome);

        // Reset RNG
        let mut rng = StdRng::seed_from_u64(42);

        // Probability 1.0 should always choose parent A
        let crossover = Crossover::Uniform { probability: 1.0 };
        let child_genome = crossover.crossover(&mut rng, &parent_a, &parent_b);
        assert_eq!(child_genome, parent_a.genome);
    }
}
