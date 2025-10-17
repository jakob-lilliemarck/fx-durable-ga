use super::Population;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct Schedule {
    pub max_evaluations: u32,
    pub population_size: u32,
    pub selection_interval: u32,
}

#[derive(Debug, PartialEq)]
pub enum ScheduleDecision {
    /// Wait - not ready to breed yet
    Wait,
    /// Breed new genotypes with the specified parameters
    Breed {
        num_offspring: usize,
        next_generation_id: i32,
    },
    /// Terminate the optimization (max generations/evaluations reached)
    Terminate,
}

impl Schedule {
    /// Create a generational schedule (breeds entire population each generation)
    pub fn generational(max_generations: u32, population_size: u32) -> Self {
        Self {
            max_evaluations: max_generations * population_size,
            population_size,
            selection_interval: population_size, // Breed entire population each time
        }
    }

    /// Create a rolling schedule (breeds in smaller batches)
    pub fn rolling(max_evaluations: u32, population_size: u32, selection_interval: u32) -> Self {
        Self {
            max_evaluations,
            population_size,
            selection_interval,
        }
    }

    /// Determine what action to take based on current population state
    pub(crate) fn should_breed(&self, population: &Population) -> ScheduleDecision {
        // Guard: Reached evaluation budget - terminate
        if population.evaluated_genotypes >= (self.max_evaluations as i64) {
            return ScheduleDecision::Terminate;
        }

        // Guard: Population busy, can't breed yet - wait
        // For generational: wait while live_genotypes > 0 (current generation still evaluating)
        // For rolling: wait while not enough slots available for new batch
        if population.live_genotypes > (self.population_size - self.selection_interval) as i64 {
            return ScheduleDecision::Wait;
        }

        // Happy path: breed new genotypes
        // For generational: breed entire new population (selection_interval == population_size)
        // For rolling: breed small batch to fill available slots (selection_interval < population_size)
        ScheduleDecision::Breed {
            num_offspring: self.selection_interval as usize,
            next_generation_id: population.current_generation + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn create_test_population(
        current_generation: i32,
        live_genotypes: i64,
        evaluated_genotypes: i64,
        best_fitness: Option<f64>,
    ) -> Population {
        Population {
            request_id: Uuid::now_v7(),
            current_generation,
            live_genotypes,
            evaluated_genotypes,
            best_fitness,
        }
    }

    #[test]
    fn generational_constructor_sets_correct_parameters() {
        let schedule = Schedule::generational(5, 100);

        assert_eq!(schedule.max_evaluations, 500); // 5 * 100
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

    #[test]
    fn generational_should_wait_when_generation_still_evaluating() {
        let schedule = Schedule::generational(5, 100);
        let population = create_test_population(1, 50, 100, Some(0.5));

        // live_genotypes > 0, so should wait
        assert_eq!(schedule.should_breed(&population), ScheduleDecision::Wait);
    }

    #[test]
    fn generational_should_breed_when_generation_complete() {
        let schedule = Schedule::generational(5, 100);
        let population = create_test_population(1, 0, 200, Some(0.6));

        // live_genotypes == 0 and under budget, so should breed
        let decision = schedule.should_breed(&population);
        assert_eq!(
            decision,
            ScheduleDecision::Breed {
                num_offspring: 100,
                next_generation_id: 2,
            }
        );
    }

    #[test]
    fn generational_should_terminate_when_budget_reached() {
        let schedule = Schedule::generational(5, 100); // max_evaluations = 500
        let population = create_test_population(5, 0, 500, Some(0.8));

        // evaluated_genotypes >= max_evaluations, so should terminate
        assert_eq!(
            schedule.should_breed(&population),
            ScheduleDecision::Terminate
        );
    }

    #[test]
    fn rolling_should_wait_when_not_enough_slots() {
        let schedule = Schedule::rolling(1000, 100, 20);
        let population = create_test_population(3, 81, 300, Some(0.7));

        // live_genotypes (81) > population_size - selection_interval (80), so should wait
        assert_eq!(schedule.should_breed(&population), ScheduleDecision::Wait);
    }

    #[test]
    fn rolling_should_breed_when_slots_available() {
        let schedule = Schedule::rolling(1000, 100, 20);
        let population = create_test_population(3, 80, 300, Some(0.7));

        // live_genotypes (80) <= population_size - selection_interval (80), so should breed
        let decision = schedule.should_breed(&population);
        assert_eq!(
            decision,
            ScheduleDecision::Breed {
                num_offspring: 20,
                next_generation_id: 4,
            }
        );
    }

    #[test]
    fn rolling_should_terminate_when_budget_reached() {
        let schedule = Schedule::rolling(1000, 100, 20);
        let population = create_test_population(10, 60, 1000, Some(0.9));

        // evaluated_genotypes >= max_evaluations, so should terminate
        assert_eq!(
            schedule.should_breed(&population),
            ScheduleDecision::Terminate
        );
    }

    #[test]
    fn rolling_with_selection_interval_equals_population_behaves_like_generational() {
        let schedule = Schedule::rolling(500, 100, 100); // selection_interval == population_size

        // Should wait while any genotypes are evaluating (like generational)
        let population = create_test_population(1, 1, 150, Some(0.5));
        assert_eq!(schedule.should_breed(&population), ScheduleDecision::Wait);

        // Should breed entire population when none are evaluating (like generational)
        let population = create_test_population(2, 0, 250, Some(0.7));
        let decision = schedule.should_breed(&population);
        assert_eq!(
            decision,
            ScheduleDecision::Breed {
                num_offspring: 100,
                next_generation_id: 3,
            }
        );
    }
}
