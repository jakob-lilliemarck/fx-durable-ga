use serde::{Deserialize, Serialize};

/// Genetic algorithm evolution strategies.
///
/// IMPORTANT: Both strategies increment generation IDs for each breeding batch.
/// This ensures atomic generation creation and prevents race conditions when
/// multiple workers try to breed simultaneously. Rolling strategies create
/// "micro-generations" where each batch gets a unique generation ID, rather
/// than keeping all offspring in generation 1.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub enum Strategy {
    Rolling {
        /// the maaximum number of evaluations to run before termination
        max_evaluations: u32,
        /// the maximum number of genotypes under evaluation ("active" genotypes)
        population_size: u32,
        /// the number of evaluations to wait before breeding new genotypes. larger pool = more diversity
        selection_interval: u32,
        /// the number of genotypes per tournament
        tournament_size: u32,
        /// the number of genotypes to fetch from the breeding bool during selection
        sample_size: u32,
    },
    Generational {
        max_generations: u32,
        population_size: u32,
    },
}

impl Strategy {
    pub fn generational(max_generations: u32, population_size: u32) -> Self {
        Self::Generational {
            max_generations,
            population_size,
        }
    }
}
