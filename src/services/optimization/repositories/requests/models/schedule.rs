use serde::{Deserialize, Serialize};

/// Controls when new generations are bred during optimization.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Schedule {
    /// Total evaluation budget before optimization terminates.
    pub(crate) max_evaluations: u32,
    /// Maximum number of genotypes that can be active simultaneously.
    pub(crate) population_size: u32,
    /// Number of offspring created per breeding cycle.
    pub(crate) selection_interval: u32,
}

impl Schedule {
    pub fn new(max_evaluations: u32, population_size: u32, selection_interval: u32) -> Self {
        Self {
            max_evaluations,
            population_size,
            selection_interval,
        }
    }

    /// Creates a generational schedule that breeds the entire population each generation.
    pub fn generational(max_generations: u32, population_size: u32) -> Self {
        Self {
            max_evaluations: max_generations * population_size,
            population_size,
            selection_interval: population_size,
        }
    }

    /// Creates a rolling schedule that breeds in smaller batches at regular intervals.
    pub fn rolling(max_evaluations: u32, population_size: u32, selection_interval: u32) -> Self {
        Self::new(max_evaluations, population_size, selection_interval)
    }

    pub(crate) fn is_generational(&self) -> bool {
        self.population_size == self.selection_interval
    }

    pub(crate) fn population_size(&self) -> u32 {
        self.population_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generational_constructor_sets_correct_parameters() {
        let schedule = Schedule::generational(5, 100);

        assert_eq!(schedule.max_evaluations, 500);
        assert_eq!(schedule.population_size, 100);
        assert_eq!(schedule.selection_interval, 100);
    }

    #[test]
    fn rolling_constructor_sets_correct_parameters() {
        let schedule = Schedule::rolling(1000, 200, 20);

        assert_eq!(schedule.max_evaluations, 1000);
        assert_eq!(schedule.population_size, 200);
        assert_eq!(schedule.selection_interval, 20);
    }
}
