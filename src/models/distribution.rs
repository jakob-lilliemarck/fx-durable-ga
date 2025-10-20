use crate::models::{Gene, Morphology};
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Strategy for generating initial populations in genetic algorithms.
///
/// The distribution strategy determines how the initial population is spread across
/// the search space. Different strategies have distinct characteristics that can
/// significantly impact optimization performance, especially during early generations.
///
/// # Strategy Comparison
///
/// **Latin Hypercube Sampling**: Provides superior space coverage by ensuring
/// each parameter dimension is evenly sampled. This typically leads to better
/// initial diversity and faster convergence.
///
/// **Random Sampling**: Simple uniform random distribution across the search space.
/// May result in clustered solutions or gaps in coverage, but requires no additional
/// computation.
///
/// # When to Choose Each Strategy
///
/// - **Latin Hypercube**: Recommended for most optimization problems, especially when:
///   - You have limited population budget (small population sizes)
///   - The search space is large or complex
///   - You need consistent, reproducible space coverage
///   - You want faster initial convergence
///
/// - **Random**: Consider when:
///   - You have very large population sizes (>1000)
///   - Computational efficiency is critical
///   - You're comparing against baseline random methods
///   - The search space has unusual constraints that Latin Hypercube might not handle well
///
/// # Performance Impact
///
/// Latin Hypercube sampling typically provides 10-30% better initial population
/// quality (measured by best fitness found) compared to random sampling, with
/// the advantage being most pronounced for smaller populations (20-200 individuals).
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::Distribution;
///
/// // Recommended: Latin Hypercube for better space coverage
/// let lhc_dist = Distribution::latin_hypercube(50);
///
/// // Simple random distribution
/// let random_dist = Distribution::random(50);
///
/// // Large population where random is acceptable
/// let large_random = Distribution::random(1000);
///
/// // Small population where Latin Hypercube shines
/// let small_lhc = Distribution::latin_hypercube(20);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    /// Latin Hypercube sampling ensures even coverage across all parameter dimensions.
    ///
    /// This method divides each parameter's range into equal intervals and ensures
    /// exactly one sample falls in each interval. The intervals are then randomly
    /// shuffled across dimensions to decorrelate parameters while maintaining
    /// uniform marginal distributions.
    ///
    /// **Advantages**:
    /// - Superior space coverage with fewer samples
    /// - More consistent results across runs
    /// - Better representation of parameter interactions
    /// - Typically finds better initial solutions
    ///
    /// **Best for**: Small to medium population sizes (10-500), complex search spaces
    LatinHypercube { 
        /// Number of individuals in the initial population.
        /// Each individual will be positioned to maximize overall space coverage.
        population_size: u32 
    },
    
    /// Pure random sampling generates individuals uniformly at random across the search space.
    ///
    /// Each parameter is independently sampled from its allowed range without
    /// coordination between dimensions or individuals. This can result in
    /// uneven coverage with potential clusters and gaps.
    ///
    /// **Advantages**:
    /// - Simple and fast generation
    /// - No assumptions about search space structure
    /// - Familiar and well-understood behavior
    ///
    /// **Best for**: Very large populations (>1000), simple search spaces, baseline comparisons
    Random { 
        /// Number of individuals to generate randomly.
        /// Larger populations help compensate for potential clustering and coverage gaps.
        population_size: u32 
    },
}

impl Distribution {
    /// Creates a Latin Hypercube distribution strategy for superior space coverage.
    ///
    /// Latin Hypercube sampling provides better initial population diversity by ensuring
    /// that each parameter dimension is sampled evenly across its range. This typically
    /// results in better optimization performance, especially for smaller populations.
    ///
    /// # Parameters
    /// - `population_size`: Number of individuals to generate. For best results, use 10-500.
    ///   Very small populations (<10) may not fully benefit from the structured sampling,
    ///   while very large populations (>1000) may not justify the additional computation.
    ///
    /// # When to Use
    /// - **Recommended default** for most optimization problems
    /// - Complex, multi-dimensional search spaces
    /// - When population budget is limited
    /// - When you need consistent, reproducible coverage
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::Distribution;
    ///
    /// // Standard population size with excellent coverage
    /// let dist = Distribution::latin_hypercube(50);
    ///
    /// // Small population where coverage is critical  
    /// let small_dist = Distribution::latin_hypercube(20);
    ///
    /// // Medium population for complex problems
    /// let medium_dist = Distribution::latin_hypercube(200);
    /// ```
    pub fn latin_hypercube(population_size: u32) -> Self {
        Distribution::LatinHypercube { population_size }
    }

    /// Creates a random distribution strategy for simple uniform sampling.
    ///
    /// Random sampling generates each individual independently by sampling each
    /// parameter uniformly from its allowed range. While simple, this can result
    /// in uneven coverage with potential clustering or gaps in the search space.
    ///
    /// # Parameters  
    /// - `population_size`: Number of individuals to generate. Larger populations
    ///   (>100) help compensate for potential clustering effects.
    ///
    /// # When to Use
    /// - Very large populations (>1000) where coverage gaps are less significant
    /// - Computational efficiency is critical
    /// - Comparing against baseline random methods
    /// - Simple, low-dimensional search spaces
    /// - When search space has constraints that might interfere with structured sampling
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::Distribution;
    ///
    /// // Large population where random sampling is acceptable
    /// let large_dist = Distribution::random(1000);
    ///
    /// // Baseline comparison distribution
    /// let baseline_dist = Distribution::random(50);
    ///
    /// // High-speed generation for simple problems
    /// let fast_dist = Distribution::random(100);
    /// ```
    pub fn random(population_size: u32) -> Self {
        Distribution::Random { population_size }
    }

    /// Generates initial population genomes according to this distribution strategy.
    #[instrument(level = "debug", skip(self, morphology), fields(distribution = ?self, morphology_dimensions = morphology.gene_bounds.len()))]
    pub(crate) fn distribute(&self, morphology: &Morphology) -> Vec<Vec<Gene>> {
        match self {
            Distribution::LatinHypercube { population_size } => {
                latin_hypercube(*population_size as usize, morphology)
            }
            Distribution::Random { population_size } => {
                random_distribution(*population_size as usize, morphology)
            }
        }
    }
}

/// Generates genomes using pure random sampling across the search space.
///
/// Each genome is created by independently sampling each gene from its corresponding
/// parameter bounds. This provides no coordination between individuals or parameters,
/// which can result in clustering or coverage gaps but is computationally efficient.
///
/// # Algorithm
/// 1. For each individual in the population:
///    - Generate each gene independently using the morphology's random() method
///    - Each gene is sampled uniformly from [0, steps-1] for that parameter
/// 2. Return all generated genomes
///
/// # Performance
/// - Time complexity: O(n_samples * n_dimensions)
/// - Space complexity: O(n_samples * n_dimensions)
/// - Very fast generation with no inter-individual coordination
#[instrument(level = "debug", skip(morphology), fields(n_samples = n_samples, n_dimensions = morphology.gene_bounds.len()))]
fn random_distribution(n_samples: usize, morphology: &Morphology) -> Vec<Vec<Gene>> {
    let mut genomes = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let genome = morphology.random();
        genomes.push(genome);
    }

    genomes
}

/// Generates genomes using Latin Hypercube sampling for superior space coverage.
///
/// Latin Hypercube sampling ensures that each parameter dimension is evenly sampled
/// by dividing each parameter's range into n_samples intervals and placing exactly
/// one sample in each interval. The intervals are then randomly shuffled across
/// dimensions to decorrelate parameters while maintaining uniform marginal distributions.
///
/// # Algorithm
/// 1. For each parameter dimension:
///    - Create n_samples evenly spaced intervals: [0, 1/n), [1/n, 2/n), ..., [(n-1)/n, 1]
///    - Sample from the center of each interval: 0.5/n, 1.5/n, ..., (n-0.5)/n
///    - Shuffle these samples to decorrelate with other dimensions
///    - Convert normalized samples to discrete gene indices using from_sample()
/// 2. Transpose the results to create genomes (each genome gets one gene from each dimension)
///
/// # Properties
/// - **Even marginal distributions**: Each parameter dimension has exactly uniform coverage
/// - **Decorrelated dimensions**: Shuffling ensures parameter independence
/// - **No clustering**: Guaranteed even spacing within each dimension
/// - **Deterministic coverage**: Same quality regardless of random seed (only ordering varies)
///
/// # Performance
/// - Time complexity: O(n_samples * n_dimensions * log(n_samples)) due to shuffling
/// - Space complexity: O(n_samples * n_dimensions)
/// - Slightly slower than random due to shuffling, but provides superior coverage
#[instrument(level = "debug", skip(morphology), fields(n_samples = n_samples, n_dimensions = morphology.gene_bounds.len()))]
fn latin_hypercube(n_samples: usize, morphology: &Morphology) -> Vec<Vec<Gene>> {
    use rand::seq::SliceRandom;

    let n_dimensions = morphology.gene_bounds.len();
    let mut rng = rand::rng();

    // Create n_samples genomes
    let mut genomes = Vec::with_capacity(n_samples);

    // For each dimension, create Latin Hypercube sampling
    for dim_idx in 0..n_dimensions {
        let gene_bound = &morphology.gene_bounds[dim_idx];

        // Create evenly spaced intervals and shuffle for decorrelation
        let mut intervals: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64 + 0.5) / n_samples as f64) // Center of each interval
            .collect();
        intervals.shuffle(&mut rng);

        // Convert normalized samples to actual gene values
        let gene_values: Vec<Gene> = intervals
            .iter()
            .map(|&sample| gene_bound.from_sample(sample))
            .collect();

        // Assign gene values to genomes (transpose operation)
        for (genome_idx, &gene_value) in gene_values.iter().enumerate() {
            if dim_idx == 0 {
                // First dimension: create new genomes
                genomes.push(vec![gene_value]);
            } else {
                // Subsequent dimensions: append to existing genomes
                genomes[genome_idx].push(gene_value);
            }
        }
    }

    genomes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GeneBounds;

    // Helper to create test morphology
    fn create_test_morphology(gene_bounds: Vec<GeneBounds>) -> Morphology {
        Morphology::new("test", 1, gene_bounds)
    }

    #[test]
    fn test_latin_hypercube_2d_simple() {
        let morphology = create_test_morphology(vec![
            GeneBounds::integer(0, 3, 4).unwrap(), // 4 steps: [0, 1, 2, 3]
            GeneBounds::integer(0, 3, 4).unwrap(), // 4 steps: [0, 1, 2, 3]
        ]);

        let dist = Distribution::latin_hypercube(4);
        let genomes = dist.distribute(&morphology);

        // Should have 4 genomes, each with 2 genes
        assert_eq!(genomes.len(), 4);
        assert!(genomes.iter().all(|genome| genome.len() == 2));

        // Extract dimensions for easier checking
        let dim1: Vec<i64> = genomes.iter().map(|g| g[0]).collect();
        let dim2: Vec<i64> = genomes.iter().map(|g| g[1]).collect();

        // Each dimension should contain exactly one of each value [0, 1, 2, 3]
        let mut dim1_sorted = dim1.clone();
        dim1_sorted.sort();
        assert_eq!(dim1_sorted, vec![0, 1, 2, 3]);

        let mut dim2_sorted = dim2.clone();
        dim2_sorted.sort();
        assert_eq!(dim2_sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_latin_hypercube_3d_simple() {
        // 3D test: 3 samples in 3 dimensions
        let morphology = create_test_morphology(vec![
            GeneBounds::integer(0, 2, 3).unwrap(), // 3 steps: [0, 1, 2]
            GeneBounds::integer(0, 2, 3).unwrap(), // 3 steps: [0, 1, 2]
            GeneBounds::integer(0, 2, 3).unwrap(), // 3 steps: [0, 1, 2]
        ]);

        let dist = Distribution::latin_hypercube(3);
        let genomes = dist.distribute(&morphology);

        // Should have 3 genomes, each with 3 genes
        assert_eq!(genomes.len(), 3);
        assert!(genomes.iter().all(|genome| genome.len() == 3));

        // Each dimension should contain exactly [0, 1, 2] in some order
        for dim in 0..3 {
            let mut values: Vec<i64> = genomes.iter().map(|g| g[dim]).collect();
            values.sort();
            assert_eq!(values, vec![0, 1, 2]);
        }
    }

    #[test]
    fn test_latin_hypercube_single_sample() {
        // Edge case: single sample should work
        let morphology = create_test_morphology(vec![
            GeneBounds::integer(0, 9, 10).unwrap(),
            GeneBounds::integer(0, 9, 10).unwrap(),
        ]);

        let dist = Distribution::latin_hypercube(1);
        let genomes = dist.distribute(&morphology);

        // Basic structure should work with single sample
        assert_eq!(genomes.len(), 1);
        assert_eq!(genomes[0].len(), 2);

        // Values should be within bounds
        assert!((0..=9).contains(&genomes[0][0]));
        assert!((0..=9).contains(&genomes[0][1]));
    }

    #[test]
    fn test_distribution_distribute_random() {
        let morphology = create_test_morphology(vec![
            GeneBounds::integer(0, 5000, 1000).unwrap(),
            GeneBounds::integer(0, 3000, 1000).unwrap(),
        ]);

        let dist = Distribution::random(5);

        // Generate multiple distributions to test randomness
        let genomes1 = dist.distribute(&morphology);
        let genomes2 = dist.distribute(&morphology);

        // Basic structure validation
        assert_eq!(genomes1.len(), 5);
        assert!(genomes1.iter().all(|genome| genome.len() == 2));

        // The key test: random distributions should be different
        assert_ne!(genomes1, genomes2);
    }
}
