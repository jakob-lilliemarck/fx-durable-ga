use crate::models::Genotype;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// Pure function for single roulette wheel spin
fn spin_roulette(
    candidates: &[(Genotype, f64)],
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
        let candidates = vec![
            (
                create_test_genotype("00000000-0000-0000-0000-000000000001"),
                0.1,
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000002"),
                0.3,
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000003"),
                0.6,
            ),
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
        assert!((proportion_0 - 0.1).abs() < 0.05);

        // Expect 30% with tolerance
        assert!((proportion_1 - 0.3).abs() < 0.05);

        // Expect 60% with tolerance
        assert!((proportion_2 - 0.6).abs() < 0.05);
    }

    #[test]
    fn it_always_selects_single_candidate() {
        let candidates = vec![(
            create_test_genotype("00000000-0000-0000-0000-000000000001"),
            1.0,
        )];
        let mut rng = rand::rng();

        for _ in 0..3 {
            let selected_idx = spin_roulette(&candidates, 1.0, 0.0, &mut rng).unwrap();
            assert_eq!(selected_idx, 0);
        }
    }

    #[test]
    fn it_distributes_equal_fitness_evenly() {
        let candidates = vec![
            (
                create_test_genotype("00000000-0000-0000-0000-000000000001"),
                1.0,
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000002"),
                1.0,
            ),
        ];
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
        assert!((proportion_0 - 0.5).abs() < 0.05);

        // Expect 50% with tolerance
        assert!((proportion_1 - 0.5).abs() < 0.05);
    }

    #[test]
    fn it_fails_when_total_fitness_is_incorrect() {
        let candidates = vec![(
            create_test_genotype("00000000-0000-0000-0000-000000000001"),
            0.0,
        )];
        // Pass incorrect total_fitness (higher than actual sum of 0.6)
        let total_fitness = 1.0;
        let offset = 0.0;
        let mut rng = rand::rng();

        // Since candidate has fitness 0.0 there should be no chance of the spin to be in range - this should _always_ error
        let should_err = spin_roulette(&candidates, total_fitness, offset, &mut rng);
        assert!(should_err.is_err(),);
    }
}

/// Pure function for roulette selection
fn roulette_selection(
    num_pairs: usize,
    candidates_with_fitness: Vec<(Genotype, Option<f64>)>,
    rng: &mut impl rand::Rng,
) -> Result<Vec<(usize, usize)>, SelectionError> {
    let mut parent_pairs = Vec::with_capacity(num_pairs);

    // Filter out genotypes without fitness
    let evaluated_candidates: Vec<(Genotype, f64)> = candidates_with_fitness
        .into_iter()
        .filter_map(|(genotype, fitness_opt)| fitness_opt.map(|fitness| (genotype, fitness)))
        .collect();

    if evaluated_candidates.is_empty() {
        return Err(SelectionError::NoValidParents);
    }

    // Check for negative fitness values
    if evaluated_candidates
        .iter()
        .any(|(_, fitness)| *fitness < 0.0)
    {
        return Err(SelectionError::InvalidFitnessForRoulette);
    }

    // Calculate total fitness
    let total_fitness: f64 = evaluated_candidates
        .iter()
        .map(|(_, fitness)| *fitness)
        .sum();

    if total_fitness <= 0.0 {
        return Err(SelectionError::InvalidFitnessForRoulette);
    }

    // Select parent pairs
    for _ in 0..num_pairs {
        let parent1_idx = spin_roulette(&evaluated_candidates, total_fitness, 0.0, rng)?;
        let parent2_idx = spin_roulette(&evaluated_candidates, total_fitness, 0.0, rng)?;
        parent_pairs.push((parent1_idx, parent2_idx));
    }

    Ok(parent_pairs)
}

/// Pure function for tournament selection
fn tournament_selection(
    num_pairs: usize,
    tournament_size: usize,
    candidates_with_fitness: Vec<(Genotype, Option<f64>)>,
    rng: &mut impl rand::Rng,
) -> Result<Vec<(usize, usize)>, SelectionError> {
    let mut parent_pairs = Vec::with_capacity(num_pairs);

    // Filter out genotypes without fitness
    let evaluated_candidates: Vec<(Genotype, f64)> = candidates_with_fitness
        .into_iter()
        .filter_map(|(genotype, fitness_opt)| fitness_opt.map(|fitness| (genotype, fitness)))
        .collect();

    // Check if we have enough candidates
    if evaluated_candidates.len() < tournament_size * 2 {
        return Err(SelectionError::InsufficientCandidates {
            needed: tournament_size * 2,
            available: evaluated_candidates.len(),
        });
    }

    for _ in 0..num_pairs {
        // Create indices and shuffle them
        let mut indices: Vec<usize> = (0..evaluated_candidates.len()).collect();
        indices.shuffle(rng);

        // Tournament for parent 1
        let mut parent1_idx = indices[0];
        for &idx in &indices[0..tournament_size] {
            // First iteration compares indices[0] with itself (harmless, always false)
            if evaluated_candidates[idx].1 > evaluated_candidates[parent1_idx].1 {
                parent1_idx = idx;
            }
        }

        // Tournament for parent 2
        let mut parent2_idx = indices[tournament_size];
        for &idx in &indices[tournament_size..(tournament_size * 2)] {
            // First iteration compares indices[tournament_size] with itself (harmless, always false)
            if evaluated_candidates[idx].1 > evaluated_candidates[parent2_idx].1 {
                parent2_idx = idx;
            }
        }

        parent_pairs.push((parent1_idx, parent2_idx));
    }

    Ok(parent_pairs)
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct Selector {
    pub method: SelectionMethod,
    pub sample_size: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, PartialEq, Eq))]
pub enum SelectionMethod {
    Tournament { size: usize },
    Roulette,
}

#[derive(Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum SelectionError {
    #[error("No valid parents available for selection")]
    NoValidParents,
    #[error("Insufficient candidates for tournament selection: need {needed}, got {available}")]
    InsufficientCandidates { needed: usize, available: usize },
    #[error("All candidates have zero or negative fitness for roulette selection")]
    InvalidFitnessForRoulette,
    #[error("Internal error: roulette wheel failed to select candidate")]
    RouletteSelectionFailed,
}

impl Selector {
    pub fn tournament(tournament_size: usize, sample_size: usize) -> Self {
        Self {
            method: SelectionMethod::Tournament {
                size: tournament_size,
            },
            sample_size,
        }
    }

    pub fn roulette(sample_size: usize) -> Self {
        Self {
            method: SelectionMethod::Roulette,
            sample_size,
        }
    }

    /// Select parent pairs from candidates with fitness
    pub(crate) fn select_parents(
        &self,
        num_pairs: usize,
        candidates_with_fitness: Vec<(Genotype, Option<f64>)>,
    ) -> Result<Vec<(usize, usize)>, SelectionError> {
        let mut rng = rand::rng();
        match self.method {
            SelectionMethod::Tournament { size } => {
                tournament_selection(num_pairs, size, candidates_with_fitness, &mut rng)
            }
            SelectionMethod::Roulette => {
                roulette_selection(num_pairs, candidates_with_fitness, &mut rng)
            }
        }
    }

    pub fn sample_size(&self) -> i64 {
        self.sample_size as i64
    }
}

#[cfg(test)]
mod selector_tests {
    use super::*;

    #[test]
    fn test_tournament_constructor() {
        let selector = Selector::tournament(3, 50);

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
        let tournament_selector = Selector::tournament(2, 100);
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

        let result = tournament_selection(2, 2, candidates, &mut rng);
        assert!(result.is_ok());

        let pairs = result.unwrap();
        assert_eq!(pairs.len(), 2);

        // Verify indices are valid
        for (parent1_idx, parent2_idx) in &pairs {
            assert!(*parent1_idx < 6);
            assert!(*parent2_idx < 6);
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

        let result = selector.select_parents(2, candidates);
        assert!(result.is_ok());

        let pairs = result.unwrap();
        assert_eq!(pairs.len(), 2);

        // Verify indices are valid
        for (parent1_idx, parent2_idx) in &pairs {
            assert!(*parent1_idx < 3);
            assert!(*parent2_idx < 3);
        }
    }

    #[test]
    fn test_select_parents_tournament_insufficient_candidates() {
        let selector = Selector::tournament(3, 50);

        let candidates = vec![(
            super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
            Some(1.0),
        )];

        let result = selector.select_parents(1, candidates);
        assert_eq!(
            result,
            Err(SelectionError::InsufficientCandidates {
                needed: 6,
                available: 1
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

        let result = selector.select_parents(1, candidates);
        assert_eq!(result, Err(SelectionError::NoValidParents));
    }

    #[test]
    fn test_roulette_errors_on_candidate_negative_fitness_zero() {
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

        let result = selector.select_parents(1, candidates);
        assert_eq!(result, Err(SelectionError::InvalidFitnessForRoulette));
    }

    #[test]
    fn test_roulette_errors_on_total_fitness_zero() {
        let selector = Selector::roulette(25);

        let candidates = vec![(
            super::test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
            Some(0.0),
        )];

        let result = selector.select_parents(1, candidates);
        assert_eq!(result, Err(SelectionError::InvalidFitnessForRoulette));
    }

    #[test]
    fn test_selector_clone_and_equality() {
        let selector1 = Selector::tournament(3, 50);
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
