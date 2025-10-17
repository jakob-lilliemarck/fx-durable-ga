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

        assert!(
            (proportion_0 - 0.1).abs() < 0.05,
            "Expected ~10%, got {:.1}%",
            proportion_0 * 100.0
        );
        assert!(
            (proportion_1 - 0.3).abs() < 0.05,
            "Expected ~30%, got {:.1}%",
            proportion_1 * 100.0
        );
        assert!(
            (proportion_2 - 0.6).abs() < 0.05,
            "Expected ~60%, got {:.1}%",
            proportion_2 * 100.0
        );
    }

    #[test]
    fn it_always_selects_single_candidate() {
        let candidates = vec![(
            create_test_genotype("00000000-0000-0000-0000-000000000001"),
            1.0,
        )];
        let mut rng = rand::rng();

        for _ in 0..1000 {
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

        assert!(
            (proportion_0 - 0.5).abs() < 0.05,
            "Expected ~50%, got {:.1}%",
            proportion_0 * 100.0
        );
        assert!(
            (proportion_1 - 0.5).abs() < 0.05,
            "Expected ~50%, got {:.1}%",
            proportion_1 * 100.0
        );
    }
}

/// Pure function for roulette selection
fn roulette_selection(
    num_pairs: usize,
    candidates_with_fitness: Vec<(Genotype, Option<f64>)>,
) -> Result<Vec<(usize, usize)>, SelectionError> {
    let mut rng = rand::rng();
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
        let parent1_idx = spin_roulette(&evaluated_candidates, total_fitness, 0.0, &mut rng)?;
        let parent2_idx = spin_roulette(&evaluated_candidates, total_fitness, 0.0, &mut rng)?;
        parent_pairs.push((parent1_idx, parent2_idx));
    }

    Ok(parent_pairs)
}

#[cfg(test)]
mod roulette_selection_tests {
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
    fn it_performs_roulette_selection() {
        // Test with valid candidates
        let candidates = vec![
            (
                create_test_genotype("00000000-0000-0000-0000-000000000001"),
                Some(1.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000002"),
                Some(2.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000003"),
                Some(3.0),
            ),
        ];

        let result = roulette_selection(2, candidates);
        assert!(result.is_ok());

        let pairs = result.unwrap();
        assert_eq!(pairs.len(), 2);

        // Each pair should contain two valid indices
        for (parent1_idx, parent2_idx) in &pairs {
            assert!(*parent1_idx < 3); // Valid index for 3 candidates
            assert!(*parent2_idx < 3);
        }
    }

    #[test]
    fn it_handles_no_valid_parents() {
        let candidates = vec![(
            create_test_genotype("00000000-0000-0000-0000-000000000001"),
            None,
        )];

        let result = roulette_selection(1, candidates);
        assert!(matches!(result, Err(SelectionError::NoValidParents)));
    }

    #[test]
    fn it_rejects_negative_fitness() {
        let candidates = vec![
            (
                create_test_genotype("00000000-0000-0000-0000-000000000001"),
                Some(-1.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000002"),
                Some(2.0),
            ),
        ];

        let result = roulette_selection(1, candidates);
        assert!(matches!(
            result,
            Err(SelectionError::InvalidFitnessForRoulette)
        ));
    }
}

/// Pure function for tournament selection
fn tournament_selection(
    num_pairs: usize,
    tournament_size: usize,
    candidates_with_fitness: Vec<(Genotype, Option<f64>)>,
) -> Result<Vec<(usize, usize)>, SelectionError> {
    let mut rng = rand::rng();
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
        indices.shuffle(&mut rng);

        // Tournament for parent 1
        let parent1_idx = indices[..tournament_size]
            .iter()
            .max_by(|&&a, &&b| {
                evaluated_candidates[a]
                    .1
                    .partial_cmp(&evaluated_candidates[b].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(SelectionError::NoValidParents)?
            .clone();

        // Tournament for parent 2 (from remaining candidates)
        let parent2_idx = indices[tournament_size..(tournament_size * 2)]
            .iter()
            .max_by(|&&a, &&b| {
                evaluated_candidates[a]
                    .1
                    .partial_cmp(&evaluated_candidates[b].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(SelectionError::NoValidParents)?
            .clone();

        parent_pairs.push((parent1_idx, parent2_idx));
    }

    Ok(parent_pairs)
}

#[cfg(test)]
mod tournament_selection_tests {
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
    fn it_performs_tournament_selection() {
        // Create enough candidates for tournament selection (need at least tournament_size * 2)
        let candidates = vec![
            (
                create_test_genotype("00000000-0000-0000-0000-000000000001"),
                Some(1.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000002"),
                Some(2.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000003"),
                Some(3.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000004"),
                Some(4.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000005"),
                Some(5.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000006"),
                Some(6.0),
            ),
        ];

        let result = tournament_selection(2, 2, candidates);
        assert!(result.is_ok());

        let pairs = result.unwrap();
        assert_eq!(pairs.len(), 2);

        // Verify all selected indices are valid
        for (parent1_idx, parent2_idx) in &pairs {
            assert!(*parent1_idx < 6); // Valid index for 6 candidates
            assert!(*parent2_idx < 6);
        }
    }

    #[test]
    fn it_handles_insufficient_candidates() {
        let candidates = vec![(
            create_test_genotype("00000000-0000-0000-0000-000000000001"),
            Some(1.0),
        )];

        let result = tournament_selection(1, 3, candidates); // Need 6 candidates for tournament_size=3
        assert!(matches!(
            result,
            Err(SelectionError::InsufficientCandidates {
                needed: 6,
                available: 1
            })
        ));
    }

    #[test]
    fn it_filters_candidates_without_fitness() {
        let candidates = vec![
            (
                create_test_genotype("00000000-0000-0000-0000-000000000001"),
                Some(1.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000002"),
                None,
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000003"),
                Some(3.0),
            ),
            (
                create_test_genotype("00000000-0000-0000-0000-000000000004"),
                Some(4.0),
            ),
        ];

        // Only 3 candidates have fitness, need 4 for tournament_size=2
        let result = tournament_selection(1, 2, candidates);
        assert!(matches!(
            result,
            Err(SelectionError::InsufficientCandidates {
                needed: 4,
                available: 3
            })
        ));
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct Selector {
    pub method: SelectionMethod,
    pub sample_size: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub enum SelectionMethod {
    Tournament { size: usize },
    Roulette,
}

#[derive(Debug, thiserror::Error)]
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
        match self.method {
            SelectionMethod::Tournament { size } => {
                tournament_selection(num_pairs, size, candidates_with_fitness)
            }
            SelectionMethod::Roulette => roulette_selection(num_pairs, candidates_with_fitness),
        }
    }

    pub fn sample_size(&self) -> i64 {
        self.sample_size as i64
    }
}
