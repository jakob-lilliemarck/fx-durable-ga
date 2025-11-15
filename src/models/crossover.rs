use crate::models::{Gene, Genotype};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Performs uniform crossover by selecting genes from each parent with the given probability.
#[instrument(level = "debug", skip(rng, lhs, rhs), fields(genome_length = lhs.genome().len(), probability = probability))]
fn crossover_uniform<R: Rng>(
    rng: &mut R,
    lhs: &Genotype,
    rhs: &Genotype,
    probability: f64,
) -> Vec<Gene> {
    let lhs_genome = lhs.genome();
    let rhs_genome = rhs.genome();
    lhs_genome
        .iter()
        .zip(rhs_genome.iter())
        .map(|(&lhs, &rhs)| {
            if rng.random_bool(probability) {
                lhs
            } else {
                rhs
            }
        })
        .collect()
}

/// Performs single-point crossover at the specified cut point.
#[instrument(level = "debug", skip(lhs, rhs), fields(genome_length = lhs.genome().len(), cut_point = point))]
fn crossover_single_point(lhs: &Genotype, rhs: &Genotype, point: usize) -> Vec<Gene> {
    let lhs_genome = lhs.genome();
    let rhs_genome = rhs.genome();
    let mut genome = Vec::with_capacity(lhs_genome.len());

    genome.extend_from_slice(&lhs_genome[..point]); // First part from lhs
    genome.extend_from_slice(&rhs_genome[point..]); // Second part from rhs
    genome
}

/// Crossover strategy for combining genetic material from two parents.
///
/// Crossover is a fundamental genetic algorithm operation that creates offspring by
/// combining genetic material from two parent genotypes. This enum provides different
/// strategies for how genes are selected and combined.
///
/// # Strategies
///
/// ## Uniform Crossover
/// Each gene position is independently selected from either parent based on a probability.
/// This provides fine-grained mixing of genetic material and is excellent for exploring
/// the solution space when genes are largely independent.
///
/// **When to use:**
/// - Problems where genes don't have strong positional dependencies
/// - When you want maximum genetic diversity in offspring
/// - For exploration-heavy phases of optimization
///
/// ## Single-Point Crossover
/// Selects a random cut point and takes genes before the cut from one parent and
/// genes after the cut from the other parent. This preserves larger genetic segments
/// and is ideal when genes have positional relationships.
///
/// **When to use:**
/// - Problems where adjacent genes work together (building blocks)
/// - When you want to preserve genetic segments
/// - For exploitation-focused optimization phases
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::Crossover;
///
/// // Uniform crossover with 60% chance of selecting from first parent
/// let uniform = Crossover::uniform(0.6)?;
///
/// // Single-point crossover for preserving gene segments
/// let single_point = Crossover::single_point();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Configuration Guidelines
///
/// **For exploration (early optimization):**
/// - Use `Uniform` with probability around 0.4-0.6
/// - Higher genetic diversity helps find promising regions
///
/// **For exploitation (later optimization):**
/// - Use `SinglePoint` to preserve successful gene combinations
/// - Or `Uniform` with probability closer to 0.5 for balanced mixing
///
/// **Problem-specific considerations:**
/// - Permutation problems: Often benefit from specialized crossover (consider SinglePoint)
/// - Continuous optimization: Uniform crossover typically works well
/// - Combinatorial problems: Test both strategies to see which preserves building blocks better
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Crossover {
    /// Uniform crossover selects each gene from either parent with the given probability.
    ///
    /// The `probability` parameter controls the likelihood of selecting a gene from the
    /// first parent at each position. Must be between 0.0 and 1.0 inclusive.
    ///
    /// - `probability = 0.0`: Always selects from second parent (creates clone)
    /// - `probability = 0.5`: Equal chance from either parent (maximum mixing)
    /// - `probability = 1.0`: Always selects from first parent (creates clone)
    Uniform {
        /// Probability of selecting each gene from the first parent (0.0 to 1.0)
        probability: f64,
    },
    /// Single-point crossover cuts genomes at a random point and swaps the tails.
    ///
    /// Creates offspring by taking genes [0..cut_point) from the first parent and
    /// genes [cut_point..length) from the second parent. The cut point is randomly
    /// selected from positions 1 to genome_length-1 to ensure both parents contribute.
    SinglePoint,
}

/// Error returned when attempting to create uniform crossover with invalid probability.
///
/// Uniform crossover requires a probability value between 0.0 and 1.0 inclusive.
/// Values outside this range don't make sense mathematically and will result in
/// this error being returned from [`Crossover::uniform`].
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::Crossover;
///
/// // These will return ProbabilityOutOfRangeError
/// assert!(Crossover::uniform(-0.1).is_err());
/// assert!(Crossover::uniform(1.5).is_err());
///
/// // These are valid
/// assert!(Crossover::uniform(0.0).is_ok());
/// assert!(Crossover::uniform(0.5).is_ok());
/// assert!(Crossover::uniform(1.0).is_ok());
/// ```
#[derive(Debug, thiserror::Error)]
#[error("uniform crossover probability must be between 0.0 and 1.0, got {0}")]
pub struct ProbabilityOutOfRangeError(f64);

impl Crossover {
    /// Creates a uniform crossover strategy with specified selection probability.
    ///
    /// The probability parameter determines how often genes are selected from the first parent
    /// versus the second parent during crossover. Each gene position is independently decided.
    ///
    /// # Parameters
    ///
    /// * `probability` - Likelihood of selecting each gene from the first parent (must be 0.0-1.0)
    ///   - `0.0` = Always select from second parent (effectively clones second parent)
    ///   - `0.5` = Equal probability from both parents (maximum genetic mixing)
    ///   - `1.0` = Always select from first parent (effectively clones first parent)
    ///
    /// # Returns
    ///
    /// * `Ok(Crossover)` - Valid uniform crossover strategy
    /// * `Err(ProbabilityOutOfRangeError)` - When probability is outside [0.0, 1.0] range
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fx_durable_ga::models::Crossover;
    ///
    /// // Balanced mixing - equal chance from both parents
    /// let balanced = Crossover::uniform(0.5)?;
    ///
    /// // Favor first parent - 70% chance from first parent
    /// let favor_first = Crossover::uniform(0.7)?;
    ///
    /// // Exploration-focused - slight bias for diversity
    /// let exploratory = Crossover::uniform(0.4)?;
    ///
    /// // This will fail with an error
    /// assert!(Crossover::uniform(1.5).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// Uniform crossover generates one random number per gene, making it slightly more
    /// computationally expensive than single-point crossover for large genomes. However,
    /// it often produces better genetic diversity.
    pub fn uniform(probability: f64) -> Result<Self, ProbabilityOutOfRangeError> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(ProbabilityOutOfRangeError(probability));
        }

        Ok(Self::Uniform { probability })
    }

    /// Creates a single-point crossover strategy.
    ///
    /// Single-point crossover randomly selects a cut point in the genome and creates
    /// offspring by taking genes before the cut point from the first parent and genes
    /// after the cut point from the second parent.
    ///
    /// # Behavior
    ///
    /// - Cut point is randomly selected from positions 1 to (genome_length - 1)
    /// - Both parents always contribute at least one gene
    /// - Preserves contiguous genetic segments from each parent
    /// - Only requires one random number per crossover operation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fx_durable_ga::models::Crossover;
    ///
    /// // Create single-point crossover strategy
    /// let crossover = Crossover::single_point();
    ///
    /// // Example with genomes [A,B,C,D,E] and [1,2,3,4,5]
    /// // If cut point = 2, offspring would be [A,B,3,4,5]
    /// // If cut point = 4, offspring would be [A,B,C,D,5]
    /// ```
    ///
    /// # When to Use
    ///
    /// Single-point crossover works well when:
    /// - Adjacent genes have functional relationships (building blocks)
    /// - You want to preserve successful gene combinations
    /// - Performance is critical (faster than uniform crossover)
    /// - Problem structure suggests genes work in segments
    ///
    /// Consider uniform crossover instead when genes are largely independent.
    pub fn single_point() -> Self {
        Self::SinglePoint
    }

    /// Applies the crossover operation to two parent genotypes, producing a new genome.
    #[instrument(level = "debug", skip(self, rng, lhs, rhs), fields(crossover_type = ?self, genome_length = lhs.genome().len()))]
    pub(crate) fn apply<R: Rng>(&self, rng: &mut R, lhs: &Genotype, rhs: &Genotype) -> Vec<Gene> {
        match self {
            Self::Uniform { probability } => crossover_uniform(rng, lhs, rhs, *probability),
            Self::SinglePoint => {
                let point = rng.random_range(1..lhs.genome().len()); // Cut point
                crossover_single_point(lhs, rhs, point)
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
            assert!(gene == parent_a.genome()[i] || gene == parent_b.genome()[i]);
        }
    }

    #[test]
    fn it_performs_single_point_crossover() {
        let parent_a = create_test_genotype(vec![1, 2, 3, 4, 5]);
        let parent_b = create_test_genotype(vec![6, 7, 8, 9, 10]);

        let child_genome = crossover_single_point(&parent_a, &parent_b, 1);
        assert_eq!(child_genome, vec![1, 7, 8, 9, 10]);

        let child_genome = crossover_single_point(&parent_a, &parent_b, 2);
        assert_eq!(child_genome, vec![1, 2, 8, 9, 10]);

        let child_genome = crossover_single_point(&parent_a, &parent_b, 3);
        assert_eq!(child_genome, vec![1, 2, 3, 9, 10]);

        let child_genome = crossover_single_point(&parent_a, &parent_b, 4);
        assert_eq!(child_genome, vec![1, 2, 3, 4, 10]);
    }

    #[test]
    fn it_handles_single_point_crossover_via_enum() {
        let mut rng = StdRng::seed_from_u64(42);
        let parent_a = create_test_genotype(vec![1, 2, 3, 4, 5]);
        let parent_b = create_test_genotype(vec![6, 7, 8, 9, 10]);

        let crossover = Crossover::SinglePoint;
        let child = crossover.apply(&mut rng, &parent_a, &parent_b);

        // Verify basic properties
        assert_eq!(child.len(), 5);

        // Verify there's exactly one transition point
        let mut transitions = 0;
        let parent_a_genome = parent_a.genome();
        let parent_b_genome = parent_b.genome();
        for i in 1..child.len() {
            let prev_from_a = child[i - 1] == parent_a_genome[i - 1];
            let curr_from_a = child[i] == parent_a_genome[i];

            if prev_from_a != curr_from_a {
                transitions += 1;
            }
        }

        // Single point crossover should have exactly 1 transition
        assert_eq!(transitions, 1);

        // All genes should come from one parent or the other
        for (i, &gene) in child.iter().enumerate() {
            assert!(gene == parent_a_genome[i] || gene == parent_b_genome[i]);
        }
    }

    #[test]
    fn it_handles_uniform_crossover_extreme_probabilities() {
        let mut rng = StdRng::seed_from_u64(42);

        let parent_a = create_test_genotype(vec![1, 2, 3]);
        let parent_b = create_test_genotype(vec![4, 5, 6]);

        // Probability 0.0 should always choose parent B
        let crossover = Crossover::Uniform { probability: 0.0 };
        let child_genome = crossover.apply(&mut rng, &parent_a, &parent_b);
        assert_eq!(child_genome, parent_b.genome());

        // Reset RNG
        let mut rng = StdRng::seed_from_u64(42);

        // Probability 1.0 should always choose parent A
        let crossover = Crossover::Uniform { probability: 1.0 };
        let child_genome = crossover.apply(&mut rng, &parent_a, &parent_b);
        assert_eq!(child_genome, parent_a.genome());
    }

    #[test]
    fn it_validates_uniform_crossover_probability() {
        // Test invalid probabilities (line 45)
        let result = Crossover::uniform(-0.1);
        assert!(result.is_err());

        let result = Crossover::uniform(1.5);
        assert!(result.is_err());

        // Test valid probabilities work
        let result = Crossover::uniform(0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn it_creates_single_point_crossover() {
        // Test the constructor (lines 51-53)
        let crossover = Crossover::single_point();

        // Verify it actually works
        let mut rng = StdRng::seed_from_u64(42);
        let parent_a = create_test_genotype(vec![1, 2, 3]);
        let parent_b = create_test_genotype(vec![4, 5, 6]);

        let child = crossover.apply(&mut rng, &parent_a, &parent_b);
        assert_eq!(child.len(), 3);
    }
}
