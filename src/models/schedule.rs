use super::Population;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Controls when new generations are bred during optimization.
///
/// Schedule determines how the genetic algorithm progresses through generations,
/// supporting both traditional generational and continuous rolling strategies.
///
/// # Configuration Parameters
///
/// - `max_evaluations`: Total number of genotype evaluations before termination
/// - `population_size`: Maximum number of genotypes active at any time
/// - `selection_interval`: Number of offspring bred per breeding cycle
///
/// # Breeding Strategies
///
/// **Generational**: Breeds entire population at once, waits for all to complete.
/// Use when evaluations are fast or you need synchronized generations.
///
/// **Rolling**: Breeds smaller batches continuously as slots become available.
/// Use when evaluations are slow or you want faster iteration.
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::Schedule;
///
/// // Traditional generational approach: 10 generations of 50 genotypes each
/// let generational = Schedule::generational(10, 50);
///
/// // Rolling approach: breed 10 genotypes whenever 10 slots free up
/// let rolling = Schedule::rolling(500, 50, 10);
///
/// // Fast rolling: breed single genotypes continuously
/// let continuous = Schedule::rolling(1000, 100, 1);
/// ```
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct Schedule {
    /// Total evaluation budget before optimization terminates.
    /// Higher values allow more exploration but take longer to complete.
    pub max_evaluations: u32,
    /// Maximum number of genotypes that can be active simultaneously.
    /// Larger populations explore more diverse solutions but use more resources.
    pub population_size: u32,
    /// Number of offspring created per breeding cycle.
    /// Smaller values provide faster feedback, larger values are more efficient.
    pub selection_interval: u32,
}

/// Decision about what action to take based on current population state.
///
/// Returned by [`Schedule::should_breed`] to indicate the next optimization step.
/// Used internally by the genetic algorithm engine to coordinate breeding cycles.
#[derive(Debug, PartialEq)]
pub enum ScheduleDecision {
    /// Population is at capacity or conditions not met for breeding.
    /// The algorithm should continue evaluating existing genotypes.
    Wait,
    /// Ready to create new offspring with the specified parameters.
    Breed {
        /// Number of offspring to create in this breeding cycle
        num_offspring: usize,
        /// Generation ID to assign to the new offspring
        next_generation_id: i32,
    },
    /// Evaluation budget exhausted, optimization should stop.
    /// No more genotypes will be bred or evaluated.
    Terminate,
}

impl Schedule {
    /// Creates a generational schedule that breeds the entire population each generation.
    ///
    /// In generational mode, the algorithm waits for all genotypes in a generation
    /// to complete evaluation before breeding the next generation. This provides
    /// clear generational boundaries and ensures fair selection pressure.
    ///
    /// # Parameters
    ///
    /// - `max_generations`: Number of generations to evolve
    /// - `population_size`: Number of genotypes per generation
    ///
    /// Total evaluations will be `max_generations * population_size`.
    ///
    /// # When to Use
    ///
    /// - Fast evaluation functions (< 1 second per genotype)
    /// - When you need synchronized generations for analysis
    /// - Traditional genetic algorithm behavior
    /// - Batch processing scenarios
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fx_durable_ga::models::Schedule;
    ///
    /// // Small test run: 5 generations of 20 genotypes = 100 total evaluations
    /// let test_schedule = Schedule::generational(5, 20);
    ///
    /// // Production run: 50 generations of 100 genotypes = 5000 evaluations
    /// let production_schedule = Schedule::generational(50, 100);
    /// ```
    pub fn generational(max_generations: u32, population_size: u32) -> Self {
        Self {
            max_evaluations: max_generations * population_size,
            population_size,
            selection_interval: population_size, // Breed entire population each time
        }
    }

    /// Creates a rolling schedule that breeds in smaller batches at regular intervals.
    ///
    /// In rolling mode, new genotypes are bred whenever slots become available,
    /// providing continuous optimization without waiting for full generations.
    /// This enables faster feedback and more efficient resource utilization.
    ///
    /// # Parameters
    ///
    /// - `max_evaluations`: Total evaluation budget
    /// - `population_size`: Maximum concurrent genotypes
    /// - `selection_interval`: Genotypes bred per cycle
    ///
    /// # When to Use
    ///
    /// - Slow evaluation functions (> 10 seconds per genotype)
    /// - Limited computational resources
    /// - When you want faster iteration and feedback
    /// - Distributed or asynchronous evaluation
    ///
    /// # Parameter Guidelines
    ///
    /// **Selection Interval Size:**
    /// - Small (1-5): Fastest feedback, more breeding overhead
    /// - Medium (10-25): Balanced performance and efficiency
    /// - Large (50+): Approaching generational behavior
    ///
    /// **Population to Interval Ratio:**
    /// - High ratio (10:1): More diversity, slower convergence
    /// - Low ratio (2:1): Faster convergence, less exploration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fx_durable_ga::models::Schedule;
    ///
    /// // Fast iteration: breed 5 genotypes whenever 5 slots open
    /// let fast_rolling = Schedule::rolling(1000, 50, 5);
    ///
    /// // Balanced: breed 20% of population at a time
    /// let balanced = Schedule::rolling(2000, 100, 20);
    ///
    /// // Conservative: breed 10% with high diversity
    /// let conservative = Schedule::rolling(5000, 200, 20);
    ///
    /// // Continuous: breed single genotypes as soon as possible
    /// let continuous = Schedule::rolling(1000, 100, 1);
    /// ```
    pub fn rolling(max_evaluations: u32, population_size: u32, selection_interval: u32) -> Self {
        Self {
            max_evaluations,
            population_size,
            selection_interval,
        }
    }

    /// Determines what action to take based on current population state.
    #[instrument(level = "debug", skip(self, population), fields(evaluated = population.evaluated_genotypes, live = population.live_genotypes, generation = population.current_generation, max_evaluations = self.max_evaluations))]
    pub(crate) fn should_breed(&self, population: &Population) -> ScheduleDecision {
        // Check if evaluation budget is exhausted
        if population.evaluated_genotypes >= (self.max_evaluations as i64) {
            return ScheduleDecision::Terminate;
        }

        // Check if population has capacity for new genotypes
        // Generational: wait while any genotypes are still evaluating
        // Rolling: wait while not enough slots available for new batch
        if population.live_genotypes > (self.population_size - self.selection_interval) as i64 {
            return ScheduleDecision::Wait;
        }

        // Ready to breed new genotypes
        // Generational: breeds entire population at once
        // Rolling: breeds smaller batches continuously
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
        min_fitness: Option<f64>,
        max_fitness: Option<f64>,
    ) -> Population {
        Population {
            request_id: Uuid::now_v7(),
            current_generation,
            live_genotypes,
            evaluated_genotypes,
            min_fitness,
            max_fitness,
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
        let population = create_test_population(1, 50, 100, Some(0.5), Some(0.0));

        // live_genotypes > 0, so should wait
        assert_eq!(schedule.should_breed(&population), ScheduleDecision::Wait);
    }

    #[test]
    fn generational_should_breed_when_generation_complete() {
        let schedule = Schedule::generational(5, 100);
        let population = create_test_population(1, 0, 200, Some(0.6), Some(0.0));

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
        let population = create_test_population(5, 0, 500, Some(0.8), Some(0.0));

        // evaluated_genotypes >= max_evaluations, so should terminate
        assert_eq!(
            schedule.should_breed(&population),
            ScheduleDecision::Terminate
        );
    }

    #[test]
    fn rolling_should_wait_when_not_enough_slots() {
        let schedule = Schedule::rolling(1000, 100, 20);
        let population = create_test_population(3, 81, 300, Some(0.7), Some(0.0));

        // live_genotypes (81) > population_size - selection_interval (80), so should wait
        assert_eq!(schedule.should_breed(&population), ScheduleDecision::Wait);
    }

    #[test]
    fn rolling_should_breed_when_slots_available() {
        let schedule = Schedule::rolling(1000, 100, 20);
        let population = create_test_population(3, 80, 300, Some(0.7), Some(0.0));

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
        let population = create_test_population(10, 60, 1000, Some(0.9), Some(0.0));

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
        let population = create_test_population(1, 1, 150, Some(0.5), Some(0.0));
        assert_eq!(schedule.should_breed(&population), ScheduleDecision::Wait);

        // Should breed entire population when none are evaluating (like generational)
        let population = create_test_population(2, 0, 250, Some(0.7), Some(0.0));
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
