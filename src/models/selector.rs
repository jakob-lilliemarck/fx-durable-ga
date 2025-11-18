//! Parent selection strategies for genetic algorithms.
//!
//! This module provides configurable selection methods that determine how parent
//! pairs are chosen from the candidate population for breeding operations. The
//! selection strategy significantly affects the evolutionary dynamics and convergence
//! behavior of the genetic algorithm.
//!
//! # Selection Methods
//!
//! ## Tournament Selection
//!
//! Tournament selection runs competitions between randomly chosen candidates,
//! selecting the fittest from each group. This method provides:
//!
//! - **Consistent selection pressure** regardless of fitness distribution
//! - **Tunable pressure** via tournament size parameter
//! - **Robustness** to fitness scaling and outliers
//! - **Computational efficiency** with O(k) comparisons per selection
//!
//! Tournament size guidelines:
//! - **Size 2-3**: Balanced exploration and exploitation
//! - **Size 4-5**: Moderate selection pressure for steady convergence
//! - **Size 6+**: High pressure for rapid convergence (risk of premature convergence)
//!
//! ## Roulette Wheel Selection
//!
//! Roulette wheel selection chooses candidates with probability proportional
//! to their fitness values. This method provides:
//!
//! - **Proportional selection** based on exact fitness ratios
//! - **Predictable probabilities** directly reflecting fitness differences
//! - **Smooth selection pressure** without discrete competition
//!
//! Best used when fitness values represent meaningful proportional differences
//! and have reasonable dynamic range.
//!
//! # Configuration Examples
//!
//! ## Basic Usage
//!
//! ```rust
//! use fx_durable_ga::models::Selector;
//!
//! // Tournament selection with moderate pressure
//! let tournament = Selector::tournament(3, 100);
//!
//! // Roulette wheel selection
//! let roulette = Selector::roulette(100);
//! ```
//!
//! ## Problem-Specific Configurations
//!
//! ```rust
//! use fx_durable_ga::models::Selector;
//!
//! // High exploration for complex, multimodal problems
//! let exploratory = Selector::tournament(2, 200);
//!
//! // Balanced search for general optimization
//! let balanced = Selector::tournament(4, 150);
//!
//! // Strong exploitation for fine-tuning near optima
//! let exploitative = Selector::tournament(7, 80);
//!
//! // Proportional selection when fitness ratios are meaningful
//! let proportional = Selector::roulette(120);
//! ```
//!
//! # Selection Pressure Comparison
//!
//! | Method | Pressure | Best For | Considerations |
//! |--------|----------|----------|-----------------|
//! | Tournament (size 2-3) | Low-Moderate | Complex search spaces, early generations | High exploration, slower convergence |
//! | Tournament (size 4-5) | Moderate | General optimization | Balanced exploration/exploitation |
//! | Tournament (size 6+) | High | Fine-tuning, late generations | Fast convergence, premature convergence risk |
//! | Roulette | Variable | Problems with meaningful fitness ratios | Requires non-negative fitness, sensitive to scaling |
//!
//! # Sample Size Considerations
//!
//! The `sample_size` parameter affects:
//! - **Genetic diversity**: Larger samples better represent population fitness distribution
//! - **Computational cost**: More candidates require more fitness evaluations
//! - **Selection quality**: Better samples lead to more effective parent selection
//!
//! Recommended sample sizes:
//! - **50-100**: Small populations or rapid prototyping
//! - **100-200**: Standard optimization problems
//! - **200+**: Complex problems requiring high genetic diversity

use crate::models::{FitnessGoal, Genotype};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Performs a single roulette wheel spin to select a candidate index.
fn spin_roulette(
    candidates: &[(&Genotype, f64)],
    total_fitness: f64,
    offset: f64,
    rng: &mut impl rand::Rng,
) -> Result<usize, SelectionError> {
    let spin = rng.random_range(0.0..total_fitness);
    let mut cumulative = 0.0;

    for (index, (_, fitness)) in candidates.iter().enumerate() {
        cumulative += fitness + offset;
        if cumulative >= spin {
            return Ok(index);
        }
    }

    // This should never happen with proper total_fitness calculation
    Err(SelectionError::RouletteSelectionFailed)
}

#[cfg(test)]
mod spin_roulette_tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    const TOLERANCE: f64 = 0.07;

    fn create_test_genotype(id: &str) -> Genotype {
        Genotype {
            id: Uuid::parse_str(id).unwrap(),
            generated_at: Utc::now(),
            type_name: "test".to_string(),
            type_hash: 123,
            genome: vec![1, 2, 3],
            genome_hash: Genotype::compute_genome_hash(&[1, 2, 3]),
            request_id: Uuid::now_v7(),
            generation_id: 1,
        }
    }

    #[test]
    fn it_spins_the_roulette() {
        // Create candidates with fitness values 0.1, 0.3, 0.6
        let genotypes = vec![
            create_test_genotype("00000000-0000-0000-0000-000000000001"),
            create_test_genotype("00000000-0000-0000-0000-000000000002"),
            create_test_genotype("00000000-0000-0000-0000-000000000003"),
        ];
        let candidates: Vec<(&Genotype, f64)> = vec![
            (&genotypes[0], 0.1),
            (&genotypes[1], 0.3),
            (&genotypes[2], 0.6),
        ];
        let total_fitness = 1.0;
        let offset = 0.0;

        // Spin 1000 times and count selections
        let mut counts = [0; 3];
        let mut rng = rand::rng();

        for _ in 0..1000 {
            let selected_idx = spin_roulette(&candidates, total_fitness, offset, &mut rng).unwrap();
            counts[selected_idx] += 1;
        }

        // Check if proportions are within reasonable tolerance (±5%)
        let proportion_0 = counts[0] as f64 / 1000.0;
        let proportion_1 = counts[1] as f64 / 1000.0;
        let proportion_2 = counts[2] as f64 / 1000.0;

        // Expect 10% with tolerance
        assert!((proportion_0 - 0.1).abs() < TOLERANCE);

        // Expect 30% with tolerance
        assert!((proportion_1 - 0.3).abs() < TOLERANCE);

        // Expect 60% with tolerance
        assert!((proportion_2 - 0.6).abs() < TOLERANCE);
    }

    #[test]
    fn it_always_selects_single_candidate() {
        let genotype = create_test_genotype("00000000-0000-0000-0000-000000000001");
        let candidates = vec![(&genotype, 1.0)];
        let mut rng = rand::rng();

        for _ in 0..3 {
            let selected_idx = spin_roulette(&candidates, 1.0, 0.0, &mut rng).unwrap();
            assert_eq!(selected_idx, 0);
        }
    }

    #[test]
    fn it_distributes_equal_fitness_evenly() {
        let genotype_1 = create_test_genotype("00000000-0000-0000-0000-000000000001");
        let genotype_2 = create_test_genotype("00000000-0000-0000-0000-000000000002");
        let candidates = vec![(&genotype_1, 1.0), (&genotype_2, 1.0)];
        let mut counts = [0; 2];
        let mut rng = rand::rng();

        for _ in 0..1000 {
            let idx = spin_roulette(&candidates, 2.0, 0.0, &mut rng).unwrap();
            counts[idx] += 1;
        }

        // Should be roughly 50/50 (±5%)
        let proportion_0 = counts[0] as f64 / 1000.0;
        let proportion_1 = counts[1] as f64 / 1000.0;

        // Expect 50% with tolerance
        assert!((proportion_0 - 0.5).abs() < TOLERANCE);

        // Expect 50% with tolerance
        assert!((proportion_1 - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn it_fails_when_total_fitness_is_incorrect() {
        let genotype = create_test_genotype("00000000-0000-0000-0000-000000000001");
        let candidates = vec![(&genotype, 0.0)];
        let total_fitness = 1.0;
        let offset = 0.0;
        let mut rng = rand::rng();

        let should_err = spin_roulette(&candidates, total_fitness, offset, &mut rng);
        assert!(should_err.is_err(),);
    }
}

/// Selects parent pairs using fitness-proportionate roulette wheel selection.
#[instrument(level = "debug", skip(candidates_with_fitness, rng), fields(num_pairs = num_pairs, num_candidates = candidates_with_fitness.len()))]
fn roulette_selection<'a>(
    num_pairs: usize,
    candidates_with_fitness: &'a [(Genotype, Option<f64>)],
    goal: &FitnessGoal,
    rng: &mut impl rand::Rng,
) -> Result<Vec<(&'a Genotype, &'a Genotype)>, SelectionError> {
    let mut parent_pairs = Vec::with_capacity(num_pairs);

    // Filter out genotypes without fitness and calculate sum
    let raw_fitness_pairs: Vec<(&Genotype, f64)> = candidates_with_fitness
        .iter()
        .filter_map(|(genotype, fitness_opt)| fitness_opt.map(|fitness| (genotype, fitness)))
        .collect();

    if raw_fitness_pairs.is_empty() {
        return Err(SelectionError::NoValidParents);
    }

    // Transform fitness values for roulette wheel based on goal
    let evaluated_candidates: Vec<(&Genotype, f64)> = match goal {
        FitnessGoal::Maximize { .. } => {
            // Find the minimum value
            let min_fitness = raw_fitness_pairs
                .iter()
                .map(|(_, fitness)| *fitness)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            // Figure out how much, if anything it needs to be shifted
            let shift = if min_fitness < 0.0 { -min_fitness } else { 0.0 };

            // Shift the values
            raw_fitness_pairs
                .iter()
                .map(|(genotype, fitness)| (*genotype, fitness + shift))
                .collect()
        }
        FitnessGoal::Minimize { .. } => {
            // Find the maximum fitness
            let max_fitness = raw_fitness_pairs
                .iter()
                .map(|(_, fitness)| *fitness)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            // Invert: lower fitness = higher weight
            raw_fitness_pairs
                .iter()
                .map(|(genotype, fitness)| (*genotype, max_fitness - fitness))
                .collect()
        }
    };

    // Calculate total fitness for roulette wheel
    let total_fitness: f64 = evaluated_candidates.iter().map(|(_, weight)| *weight).sum();

    if total_fitness <= 0.0 {
        return Err(SelectionError::InvalidFitnessForRoulette);
    }

    // Select parent pairs
    for _ in 0..num_pairs {
        let parent1_idx = spin_roulette(&evaluated_candidates, total_fitness, 0.0, rng)?;
        let parent2_idx = spin_roulette(&evaluated_candidates, total_fitness, 0.0, rng)?;
        parent_pairs.push((
            evaluated_candidates[parent1_idx].0,
            evaluated_candidates[parent2_idx].0,
        ));
    }

    Ok(parent_pairs)
}

/// Selects parent pairs using tournament selection with the given tournament size.
#[instrument(level = "debug", skip(candidates_with_fitness, rng), fields(num_pairs = num_pairs, tournament_size = tournament_size, num_candidates = candidates_with_fitness.len()))]
fn tournament_selection<'a>(
    num_pairs: usize,
    tournament_size: usize,
    candidates_with_fitness: &'a [(Genotype, Option<f64>)],
    goal: &FitnessGoal,
    rng: &mut impl rand::Rng,
) -> Result<Vec<(&'a Genotype, &'a Genotype)>, SelectionError> {
    let mut parent_pairs = Vec::with_capacity(num_pairs);

    // Filter out genotypes without fitness (borrowed)
    let evaluated_candidates: Vec<(&Genotype, f64)> = candidates_with_fitness
        .iter()
        .filter_map(|(genotype, fitness_opt)| fitness_opt.map(|fitness| (genotype, fitness)))
        .collect();

    // Check if we have enough candidates
    if evaluated_candidates.len() < tournament_size * 2 {
        return Err(SelectionError::InvalidSampleSize {
            min_required: tournament_size * 2,
            provided: evaluated_candidates.len(),
        });
    }

    for _ in 0..num_pairs {
        // Create indices and shuffle them
        let mut indices: Vec<usize> = (0..evaluated_candidates.len()).collect();
        indices.shuffle(rng);

        // Tournament for parent 1
        let mut parent1_idx = indices[0];
        for &idx in &indices[0..tournament_size] {
            if goal.is_better(
                evaluated_candidates[idx].1,
                evaluated_candidates[parent1_idx].1,
            ) {
                parent1_idx = idx;
            }
        }

        // Tournament for parent 2
        let mut parent2_idx = indices[tournament_size];
        for &idx in &indices[tournament_size..(tournament_size * 2)] {
            if goal.is_better(
                evaluated_candidates[idx].1,
                evaluated_candidates[parent2_idx].1,
            ) {
                parent2_idx = idx;
            }
        }

        parent_pairs.push((
            evaluated_candidates[parent1_idx].0,
            evaluated_candidates[parent2_idx].0,
        ));
    }

    Ok(parent_pairs)
}

/// Configuration for parent selection in genetic algorithms.
///
/// The selector determines how parent pairs are chosen from the candidate population
/// for breeding operations. Different selection methods bias toward higher-fitness
/// candidates to varying degrees, affecting the evolutionary pressure.
///
/// # Selection Pressure
///
/// **Tournament Selection**: Moderate to high selection pressure. Larger tournament
/// sizes increase pressure by making it more likely that high-fitness candidates win.
///
/// **Roulette Selection**: Proportional selection pressure. Candidates are chosen
/// with probability proportional to their fitness values.
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::Selector;
///
/// // Tournament selection with size 3 (moderate selection pressure)
/// let tournament_selector = Selector::tournament(3, 100);
///
/// // Roulette wheel selection (fitness-proportionate)
/// let roulette_selector = Selector::roulette(100);
///
/// // High selection pressure tournament
/// let elite_selector = Selector::tournament(7, 50);
/// ```
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Selector {
    /// The selection algorithm to use for choosing parent pairs
    pub method: SelectionMethod,
    /// Number of candidates to sample from the population for selection
    pub sample_size: usize,
}

/// Selection algorithms available for parent selection.
///
/// Each method has different characteristics and is suitable for different
/// optimization scenarios.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum SelectionMethod {
    /// Tournament selection runs competitions between randomly selected candidates.
    ///
    /// Winners are chosen based on fitness comparison. Larger tournament sizes
    /// increase selection pressure, making it more likely that high-fitness
    /// candidates are selected.
    ///
    /// **Best for**: Problems where you want consistent convergence with
    /// adjustable selection pressure. Works well with any fitness distribution.
    ///
    /// **Tournament size guidelines**:
    /// - Size 2-3: Low to moderate selection pressure, good exploration
    /// - Size 4-5: Moderate selection pressure, balanced exploration/exploitation
    /// - Size 6+: High selection pressure, strong exploitation
    Tournament {
        /// Number of candidates that compete in each tournament.
        /// Must be at least 1. Requires at least `size * 2` evaluated candidates.
        size: usize,
    },

    /// Roulette wheel selection chooses candidates with probability proportional to fitness.
    ///
    /// Also known as fitness-proportionate selection. Candidates with higher
    /// fitness values have proportionally higher chances of being selected.
    ///
    /// **Best for**: Problems where fitness values are meaningful as proportions
    /// and you want selection probability to directly reflect fitness differences.
    ///
    /// **Requirements**:
    /// - All fitness values must be non-negative
    /// - At least one candidate must have positive fitness
    /// - Works best when fitness values have reasonable range (not too extreme)
    Roulette,
}

/// Errors that can occur during parent selection.
///
/// These errors help diagnose configuration or data issues that prevent
/// successful parent selection for breeding operations.
#[derive(Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum SelectionError {
    /// No candidates with fitness values are available for selection.
    ///
    /// This occurs when all candidates in the population have `None` fitness values,
    /// meaning no candidates have been evaluated yet.
    ///
    /// **Solution**: Ensure candidates are evaluated before attempting selection.
    #[error("No valid parents available for selection")]
    NoValidParents,

    /// Invalid sample size for tournament selection configuration.
    ///
    /// Tournament selection requires `sample_size >= tournament_size * 2` because
    /// it runs two separate tournaments (one for each parent in a breeding pair).
    #[error(
        "Sample size must be >= 2 * tournament_size. Min required: {min_required}, got {provided}"
    )]
    InvalidSampleSize {
        min_required: usize,
        provided: usize,
    },

    /// Roulette selection cannot proceed due to invalid fitness values.
    ///
    /// This error occurs when:
    /// - Any candidate has negative fitness (< 0.0)
    /// - All candidates have zero fitness (total fitness = 0.0)
    ///
    /// **Solutions**:
    /// - Ensure all fitness values are non-negative
    /// - Verify that at least some candidates have positive fitness
    /// - Consider fitness normalization if needed
    /// - Switch to tournament selection if fitness scaling is problematic
    #[error("All candidates have zero or negative fitness for roulette selection")]
    InvalidFitnessForRoulette,

    /// Internal roulette wheel algorithm failure.
    ///
    /// This should not occur under normal conditions and indicates a bug
    /// in the roulette selection implementation.
    #[error("Internal error: roulette wheel failed to select candidate")]
    RouletteSelectionFailed,
}

impl Selector {
    /// Creates a tournament selector with the specified tournament size.
    ///
    /// Tournament selection randomly selects groups of candidates and chooses
    /// the fittest from each group. This provides consistent selection pressure
    /// that can be tuned via the tournament size.
    ///
    /// # Parameters
    ///
    /// * `tournament_size` - Number of candidates competing in each tournament.
    ///   Larger values increase selection pressure. Must be at least 1.
    /// * `sample_size` - Number of candidates to sample from the population.
    ///   Should be large enough to provide genetic diversity.
    ///
    /// # Selection Pressure Guide
    ///
    /// * **Size 2**: Weak selection pressure, high exploration
    /// * **Size 3-4**: Moderate selection pressure, balanced search
    /// * **Size 5-7**: Strong selection pressure, focused exploitation
    /// * **Size 8+**: Very strong selection pressure, may cause premature convergence
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fx_durable_ga::models::Selector;
    ///
    /// // Balanced exploration and exploitation
    /// let balanced = Selector::tournament(3, 100);
    ///
    /// // High exploration for complex search spaces
    /// let exploratory = Selector::tournament(2, 150);
    ///
    /// // Strong exploitation for fine-tuning
    /// let exploitative = Selector::tournament(6, 80);
    /// ```
    ///
    /// # Requirements
    ///
    /// The population must have at least `tournament_size * 2` evaluated candidates
    /// to perform selection, as each parent pair requires two separate tournaments.
    pub fn tournament(tournament_size: usize, sample_size: usize) -> Result<Self, SelectionError> {
        let min_sample_size = tournament_size * 2;
        if sample_size < min_sample_size {
            return Err(SelectionError::InvalidSampleSize {
                min_required: min_sample_size,
                provided: sample_size,
            });
        }

        Ok(Self {
            method: SelectionMethod::Tournament {
                size: tournament_size,
            },
            sample_size,
        })
    }

    /// Creates a roulette wheel selector for fitness-proportionate selection.
    ///
    /// Roulette selection chooses candidates with probability directly proportional
    /// to their fitness values. This means a candidate with fitness 6.0 is twice
    /// as likely to be selected as one with fitness 3.0.
    ///
    /// # Parameters
    ///
    /// * `sample_size` - Number of candidates to sample from the population.
    ///   Larger samples provide better representation of the fitness distribution.
    ///
    /// # When to Use
    ///
    /// Roulette selection works best when:
    /// * Fitness values represent meaningful proportional differences
    /// * You want selection probability to directly reflect fitness ratios
    /// * The fitness landscape has reasonable dynamic range (not too extreme)
    /// * You need predictable selection behavior based on fitness distributions
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fx_durable_ga::models::Selector;
    ///
    /// // Standard roulette selection
    /// let roulette = Selector::roulette(100);
    ///
    /// // Larger sample for better fitness representation
    /// let large_sample = Selector::roulette(200);
    /// ```
    ///
    /// # Fitness Requirements
    ///
    /// * All fitness values must be non-negative (≥ 0.0)
    /// * At least one candidate must have positive fitness (> 0.0)
    /// * Total fitness across all candidates must be positive
    ///
    /// # Performance Considerations
    ///
    /// Roulette selection may struggle with:
    /// * Fitness values with extreme ranges (e.g., 0.001 vs 1000.0)
    /// * Populations where most candidates have very similar fitness
    /// * Scenarios requiring rapid convergence
    pub fn roulette(sample_size: usize) -> Self {
        Self {
            method: SelectionMethod::Roulette,
            sample_size,
        }
    }

    /// Selects parent pairs from candidates using the configured selection method.
    #[instrument(level = "debug", skip(self, candidates_with_fitness), fields(method = ?self.method, num_pairs = num_pairs, num_candidates = candidates_with_fitness.len()))]
    pub(crate) fn select_parents<'a>(
        &self,
        num_pairs: usize,
        candidates_with_fitness: &'a [(Genotype, Option<f64>)],
        goal: &FitnessGoal,
    ) -> Result<Vec<(&'a Genotype, &'a Genotype)>, SelectionError> {
        let mut rng = rand::rng();
        match self.method {
            SelectionMethod::Tournament { size } => {
                tournament_selection(num_pairs, size, candidates_with_fitness, goal, &mut rng)
            }
            SelectionMethod::Roulette => {
                roulette_selection(num_pairs, candidates_with_fitness, goal, &mut rng)
            }
        }
    }

    /// Returns the configured sample size for this selector.
    ///
    /// The sample size determines how many candidates are drawn from the
    /// population before performing selection. Larger sample sizes provide
    /// better representation of the population's fitness distribution but
    /// may increase computational cost.
    ///
    /// # Returns
    ///
    /// The sample size as a signed integer for compatibility with database
    /// query operations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fx_durable_ga::models::Selector;
    ///
    /// let selector = Selector::tournament(3, 150).expect("sample_size is larger than 2 * tournament_size");
    /// assert_eq!(selector.sample_size(), 150);
    ///
    /// let roulette = Selector::roulette(75);
    /// assert_eq!(roulette.sample_size(), 75);
    /// ```
    pub fn sample_size(&self) -> i64 {
        self.sample_size as i64
    }
}

#[cfg(test)]
mod selector_tests {
    use super::*;

    #[test]
    fn test_tournament_constructor() {
        let selector = Selector::tournament(3, 50).expect("is valid");

        assert_eq!(selector.sample_size, 50);
        assert_eq!(selector.method, SelectionMethod::Tournament { size: 3 });
    }

    #[test]
    fn test_roulette_constructor() {
        let selector = Selector::roulette(25);

        assert_eq!(selector.sample_size, 25);
        assert_eq!(selector.method, SelectionMethod::Roulette);
    }

    #[test]
    fn test_sample_size_method() {
        let tournament_selector = Selector::tournament(2, 100).expect("is valid");
        assert_eq!(tournament_selector.sample_size(), 100);

        let roulette_selector = Selector::roulette(75);
        assert_eq!(roulette_selector.sample_size(), 75);
    }

    #[test]
    fn test_select_parents_tournament() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        // Test with a seed that ensures we hit the parent2_idx assignment
        let mut rng = StdRng::seed_from_u64(1);

        let candidates = vec![
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
                Some(1.0), // lowest fitness
            ),
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000002"),
                Some(2.0),
            ),
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000003"),
                Some(3.0),
            ),
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000004"),
                Some(4.0),
            ),
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000005"),
                Some(5.0),
            ),
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000006"),
                Some(6.0), // highest fitness
            ),
        ];

        let goal = &crate::models::FitnessGoal::maximize(0.9).unwrap();
        let result = tournament_selection(2, 2, &candidates, goal, &mut rng);
        assert!(result.is_ok());

        let pairs = result.unwrap();
        assert_eq!(pairs.len(), 2);

        // Verify returned refs point into candidates
        for (p1, p2) in &pairs {
            let any_match = candidates.iter().any(|(g, _)| std::ptr::eq(*p1, g))
                && candidates.iter().any(|(g, _)| std::ptr::eq(*p2, g));
            assert!(any_match);
        }
    }

    #[test]
    fn test_select_parents_roulette() {
        let selector = Selector::roulette(25);

        let candidates = vec![
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
                Some(1.0),
            ),
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000002"),
                Some(2.0),
            ),
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000003"),
                Some(3.0),
            ),
        ];

        let goal = &crate::models::FitnessGoal::maximize(0.9).unwrap();
        let result = selector.select_parents(2, &candidates, goal);
        assert!(result.is_ok());

        let pairs = result.unwrap();
        assert_eq!(pairs.len(), 2);

        for (p1, p2) in &pairs {
            let any_match = candidates.iter().any(|(g, _)| std::ptr::eq(*p1, g))
                && candidates.iter().any(|(g, _)| std::ptr::eq(*p2, g));
            assert!(any_match);
        }
    }

    #[test]
    fn test_select_parents_tournament_insufficient_candidates() {
        let selector = Selector::tournament(3, 50).expect("is valid");

        let candidates = vec![(
            super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
            Some(1.0),
        )];

        let goal = &crate::models::FitnessGoal::maximize(0.9).unwrap();
        let result = selector.select_parents(1, &candidates, goal);
        assert_eq!(
            result,
            Err(SelectionError::InvalidSampleSize {
                min_required: 6,
                provided: 1
            })
        );
    }

    #[test]
    fn test_select_parents_roulette_no_valid_parents() {
        let selector = Selector::roulette(25);

        let candidates = vec![(
            super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
            None,
        )];

        let goal = &crate::models::FitnessGoal::maximize(0.9).unwrap();
        let result = selector.select_parents(1, &candidates, goal);
        assert_eq!(result, Err(SelectionError::NoValidParents));
    }

    #[test]
    fn test_roulette_handles_negative_fitness() {
        let selector = Selector::roulette(25);
        let candidates = vec![
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
                Some(-1.0),
            ),
            (
                super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000002"),
                Some(2.0),
            ),
        ];
        let goal = &crate::models::FitnessGoal::maximize(0.9).unwrap();
        let result = selector.select_parents(1, &candidates, goal);

        // Should succeed now that we handle negative fitness
        assert!(result.is_ok());
        let pairs = result.unwrap();
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn test_roulette_errors_on_total_fitness_zero() {
        let selector = Selector::roulette(25);

        let candidates = vec![(
            super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
            Some(0.0),
        )];

        let goal = &crate::models::FitnessGoal::maximize(0.9).unwrap();
        let result = selector.select_parents(1, &candidates, goal);
        assert_eq!(result, Err(SelectionError::InvalidFitnessForRoulette));
    }

    #[test]
    fn test_selector_clone_and_equality() {
        let selector1 = Selector::tournament(3, 50).expect("is valid");
        let selector2 = selector1.clone();

        assert_eq!(selector1, selector2);

        let selector3 = Selector::roulette(25);
        assert_ne!(selector1, selector3);
    }
}

#[cfg(test)]
mod test_utilities {
    use crate::models::Genotype;
    use chrono::Utc;
    use uuid::Uuid;

    pub(super) fn create_test_genotype(id: &str) -> Genotype {
        Genotype {
            id: Uuid::parse_str(id).unwrap(),
            generated_at: Utc::now(),
            type_name: "test".to_string(),
            type_hash: 123,
            genome: vec![1, 2, 3],
            genome_hash: Genotype::compute_genome_hash(&[1, 2, 3]),
            request_id: Uuid::now_v7(),
            generation_id: 1,
        }
    }
}
