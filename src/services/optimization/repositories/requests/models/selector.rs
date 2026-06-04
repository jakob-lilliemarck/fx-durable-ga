use super::FitnessGoal;
use crate::repositories::genotypes::Genotype;
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

    Err(SelectionError::RouletteSelectionFailed)
}

/// Selects parent pairs using fitness-proportionate roulette wheel selection.
#[instrument(level = "debug", skip(candidates_with_fitness, rng), fields(num_pairs = num_pairs, num_candidates = candidates_with_fitness.len()))]
fn roulette_selection<'a>(
    num_pairs: usize,
    candidates_with_fitness: &'a [(Genotype, f64)],
    goal: &FitnessGoal,
    rng: &mut impl rand::Rng,
) -> Result<Vec<(&'a Genotype, &'a Genotype)>, SelectionError> {
    let mut parent_pairs = Vec::with_capacity(num_pairs);

    if candidates_with_fitness.is_empty() {
        return Err(SelectionError::NoValidParents);
    }

    let evaluated_candidates: Vec<(&Genotype, f64)> = match goal {
        FitnessGoal::Maximize { .. } => {
            let min_fitness = candidates_with_fitness
                .iter()
                .map(|(_, fitness)| *fitness)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            let shift = if min_fitness < 0.0 { -min_fitness } else { 0.0 };

            candidates_with_fitness
                .iter()
                .map(|(genotype, fitness)| (genotype, fitness + shift))
                .collect()
        }
        FitnessGoal::Minimize { .. } => {
            let max_fitness = candidates_with_fitness
                .iter()
                .map(|(_, fitness)| *fitness)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            candidates_with_fitness
                .iter()
                .map(|(genotype, fitness)| (genotype, max_fitness - fitness))
                .collect()
        }
    };

    let total_fitness: f64 = evaluated_candidates.iter().map(|(_, weight)| *weight).sum();

    if total_fitness <= 0.0 {
        return Err(SelectionError::InvalidFitnessForRoulette);
    }

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
    candidates_with_fitness: &'a [(Genotype, f64)],
    goal: &FitnessGoal,
    rng: &mut impl rand::Rng,
) -> Result<Vec<(&'a Genotype, &'a Genotype)>, SelectionError> {
    let mut parent_pairs = Vec::with_capacity(num_pairs);
    let number_of_candidates = candidates_with_fitness.len();

    if number_of_candidates < tournament_size * 2 {
        return Err(SelectionError::InsufficientCandidates {
            min_required: tournament_size * 2,
            provided: candidates_with_fitness.len(),
        });
    }

    for _ in 0..num_pairs {
        let mut indices: Vec<usize> = (0..number_of_candidates).collect();
        indices.shuffle(rng);

        let mut parent1_idx = indices[0];
        for &idx in &indices[0..tournament_size] {
            if goal.is_better(
                candidates_with_fitness[idx].1,
                candidates_with_fitness[parent1_idx].1,
            ) {
                parent1_idx = idx;
            }
        }

        let mut parent2_idx = indices[tournament_size];
        for &idx in &indices[tournament_size..(tournament_size * 2)] {
            if goal.is_better(
                candidates_with_fitness[idx].1,
                candidates_with_fitness[parent2_idx].1,
            ) {
                parent2_idx = idx;
            }
        }

        parent_pairs.push((
            &candidates_with_fitness[parent1_idx].0,
            &candidates_with_fitness[parent2_idx].0,
        ));
    }

    Ok(parent_pairs)
}

/// Configuration for parent selection in genetic algorithms.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Selector {
    /// The selection algorithm to use for choosing parent pairs
    pub method: SelectionMethod,
}

/// Selection algorithms available for parent selection.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum SelectionMethod {
    /// Tournament selection runs competitions between randomly selected candidates.
    Tournament {
        /// Number of candidates that compete in each tournament.
        size: usize,
    },
    /// Roulette wheel selection chooses candidates with probability proportional to fitness.
    Roulette,
}

/// Errors that can occur during parent selection.
#[derive(Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum SelectionError {
    /// No candidates with fitness values are available for selection.
    #[error("No valid parents available for selection")]
    NoValidParents,

    /// Not enough evaluated candidates for tournament selection.
    #[error(
        "Number of candidates must be >= 2 * tournament_size. Min required: {min_required}, got {provided}"
    )]
    InsufficientCandidates {
        min_required: usize,
        provided: usize,
    },

    /// All candidates have zero or negative fitness for roulette selection.
    #[error("All candidates have zero or negative fitness for roulette selection")]
    InvalidFitnessForRoulette,

    /// Internal roulette wheel algorithm failure.
    #[error("Internal error: roulette wheel failed to select candidate")]
    RouletteSelectionFailed,
}

impl Selector {
    /// Creates a tournament selector with the specified tournament size.
    pub fn tournament(tournament_size: usize) -> Self {
        Self {
            method: SelectionMethod::Tournament {
                size: tournament_size,
            },
        }
    }

    /// Creates a roulette wheel selector for fitness-proportionate selection.
    pub fn roulette() -> Self {
        Self {
            method: SelectionMethod::Roulette,
        }
    }

    /// Selects parent pairs from candidates using the configured selection method.
    #[instrument(level = "debug", skip(self, candidates_with_fitness), fields(method = ?self.method, num_pairs = num_pairs, num_candidates = candidates_with_fitness.len()))]
    pub(crate) fn select_parents<'a>(
        &self,
        num_pairs: usize,
        candidates_with_fitness: &'a [(Genotype, f64)],
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
}

#[cfg(test)]
mod spin_roulette_tests {
    use super::*;

    const TOLERANCE: f64 = 0.07;

    #[test]
    fn it_spins_the_roulette() {
        let genotypes = vec![
            test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
            test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000002"),
            test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000003"),
        ];
        let candidates: Vec<(&Genotype, f64)> = vec![
            (&genotypes[0], 0.1),
            (&genotypes[1], 0.3),
            (&genotypes[2], 0.6),
        ];
        let total_fitness = 1.0;
        let offset = 0.0;

        let mut counts = [0; 3];
        let mut rng = rand::rng();

        for _ in 0..1000 {
            let selected_idx = spin_roulette(&candidates, total_fitness, offset, &mut rng).unwrap();
            counts[selected_idx] += 1;
        }

        let proportion_0 = counts[0] as f64 / 1000.0;
        let proportion_1 = counts[1] as f64 / 1000.0;
        let proportion_2 = counts[2] as f64 / 1000.0;

        assert!((proportion_0 - 0.1).abs() < TOLERANCE);
        assert!((proportion_1 - 0.3).abs() < TOLERANCE);
        assert!((proportion_2 - 0.6).abs() < TOLERANCE);
    }

    #[test]
    fn it_always_selects_single_candidate() {
        let genotype = test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001");
        let candidates = vec![(&genotype, 1.0)];
        let mut rng = rand::rng();

        for _ in 0..3 {
            let selected_idx = spin_roulette(&candidates, 1.0, 0.0, &mut rng).unwrap();
            assert_eq!(selected_idx, 0);
        }
    }

    #[test]
    fn it_distributes_equal_fitness_evenly() {
        let genotype_1 =
            test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001");
        let genotype_2 =
            test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000002");
        let candidates = vec![(&genotype_1, 1.0), (&genotype_2, 1.0)];
        let mut counts = [0; 2];
        let mut rng = rand::rng();

        for _ in 0..1000 {
            let idx = spin_roulette(&candidates, 2.0, 0.0, &mut rng).unwrap();
            counts[idx] += 1;
        }

        let proportion_0 = counts[0] as f64 / 1000.0;
        let proportion_1 = counts[1] as f64 / 1000.0;

        assert!((proportion_0 - 0.5).abs() < TOLERANCE);
        assert!((proportion_1 - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn it_fails_when_total_fitness_is_incorrect() {
        let genotype = test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001");
        let candidates = vec![(&genotype, 0.0)];
        let total_fitness = 1.0;
        let offset = 0.0;
        let mut rng = rand::rng();

        let should_err = spin_roulette(&candidates, total_fitness, offset, &mut rng);
        assert!(should_err.is_err());
    }
}

#[cfg(test)]
mod selector_tests {
    use super::*;

    #[test]
    fn test_tournament_constructor() {
        let selector = Selector::tournament(3);
        assert_eq!(selector.method, SelectionMethod::Tournament { size: 3 });
    }

    #[test]
    fn test_roulette_constructor() {
        let selector = Selector::roulette();
        assert_eq!(selector.method, SelectionMethod::Roulette);
    }

    #[test]
    fn test_select_parents_tournament() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(1);

        let candidates: Vec<(Genotype, f64)> = vec![
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
                1.0,
            ),
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000002"),
                2.0,
            ),
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000003"),
                3.0,
            ),
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000004"),
                4.0,
            ),
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000005"),
                5.0,
            ),
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000006"),
                6.0,
            ),
        ];

        let goal = &FitnessGoal::maximize(0.9).unwrap();
        let result = tournament_selection(2, 2, &candidates, goal, &mut rng);
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
    fn test_select_parents_roulette() {
        let selector = Selector::roulette();

        let candidates = vec![
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
                1.0,
            ),
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000002"),
                2.0,
            ),
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000003"),
                3.0,
            ),
        ];

        let goal = &FitnessGoal::maximize(0.9).unwrap();
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
    fn test_select_parents_roulette_no_valid_parents() {
        let selector = Selector::roulette();

        let candidates = vec![];

        let goal = &FitnessGoal::maximize(0.9).unwrap();
        let result = selector.select_parents(1, &candidates, goal);
        assert_eq!(result, Err(SelectionError::NoValidParents));
    }

    #[test]
    fn test_roulette_handles_negative_fitness() {
        let selector = Selector::roulette();
        let candidates = vec![
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
                -1.0,
            ),
            (
                test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000002"),
                2.0,
            ),
        ];
        let goal = &FitnessGoal::maximize(0.9).unwrap();
        let result = selector.select_parents(1, &candidates, goal);

        assert!(result.is_ok());
        let pairs = result.unwrap();
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn test_roulette_errors_on_total_fitness_zero() {
        let selector = Selector::roulette();

        let candidates = vec![(
            test_utilities::create_test_genotype("00000000-0000-0000-0000-000000000001"),
            0.0,
        )];

        let goal = &FitnessGoal::maximize(0.9).unwrap();
        let result = selector.select_parents(1, &candidates, goal);
        assert_eq!(result, Err(SelectionError::InvalidFitnessForRoulette));
    }

    #[test]
    fn test_selector_clone_and_equality() {
        let selector1 = Selector::tournament(3);
        let selector2 = selector1.clone();

        assert_eq!(selector1, selector2);

        let selector3 = Selector::roulette();
        assert_ne!(selector1, selector3);
    }
}

#[cfg(test)]
mod test_utilities {
    use crate::repositories::genotypes::Genotype;
    use serde_json;
    use uuid::Uuid;

    pub(super) fn create_test_genotype(id: &str) -> Genotype {
        Genotype::new(
            "test",
            serde_json::json!([1, 2, 3]),
            Uuid::parse_str(id).unwrap(),
            Some(1),
            None,
            None,
        )
        .unwrap()
    }
}
