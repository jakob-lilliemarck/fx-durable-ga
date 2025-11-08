use super::Error;
use super::events::{
    GenotypeEvaluatedEvent, GenotypeGenerated, OptimizationRequestedEvent, RequestCompletedEvent,
    RequestTerminatedEvent,
};
use crate::models::{
    Breeder, Crossover, Distribution, Fitness, FitnessGoal, Genotype, Mutagen, Request,
    RequestConclusion, Schedule, ScheduleDecision, Selector, Terminated,
};
use crate::repositories::chainable::{Chain, FromTx, ToTx};
use crate::repositories::{genotypes, morphologies, requests};
use crate::services::lock;
use crate::services::optimization::models::{Terminator, TypeErasedEvaluator};
use fx_event_bus::Publisher;
use std::collections::{HashMap, HashSet};
use tracing::instrument;
use uuid::Uuid;

/// Genetic algorithm optimization service that manages the entire optimization lifecycle.
/// Handles request creation, population generation, genotype evaluation, and breeding.
pub struct Service {
    pub(super) locking: lock::Service,
    pub(super) requests: requests::Repository,
    pub(super) morphologies: morphologies::Repository,
    pub(super) genotypes: genotypes::Repository,
    pub(super) evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
    pub(super) max_deduplication_attempts: i32,
}

impl Service {
    /// Creates a new service builder with the required dependencies.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn builder(
        locking: lock::Service,
        requests: requests::Repository,
        morphologies: morphologies::Repository,
        genotypes: genotypes::Repository,
    ) -> super::ServiceBuilder {
        super::ServiceBuilder {
            locking,
            requests,
            morphologies,
            genotypes,
            evaluators: HashMap::new(),
            max_deduplication_attempts: 5, // Default value
        }
    }

    /// Creates a new genetic algorithm optimization request.
    ///
    /// This is the main entry point for starting genetic algorithm optimization.
    /// It configures and initiates an evolutionary process that will attempt
    /// to find optimal solutions for your problem type.
    ///
    /// # Parameters
    ///
    /// * `type_name` - Human-readable name of the type being optimized (from `Encodeable::NAME`)
    /// * `type_hash` - Unique hash identifying the type structure (from `Encodeable::HASH`)
    /// * `goal` - Optimization objective and stopping criteria
    /// * `schedule` - Controls generation timing and population lifecycle
    /// * `selector` - Parent selection strategy for breeding operations
    /// * `mutagen` - Mutation behavior including temperature and rate schedules
    /// * `crossover` - Genetic recombination method for combining parents
    /// * `distribution` - Initial population generation strategy
    ///
    /// # Optimization Goal
    ///
    /// The `goal` parameter defines when optimization should stop:
    /// - `FitnessGoal::maximize(0.95)` - Stop when fitness reaches 95% or higher
    /// - `FitnessGoal::minimize(0.1)` - Stop when fitness drops to 10% or lower
    ///
    /// # Population Management
    ///
    /// The `schedule` controls generation timing and resource allocation:
    /// - `Schedule::generational(100, 50)` - 100 generations, 50 individuals each
    /// - `Schedule::rolling(5000, 100, 10)` - 5000 total evaluations, 100 max population, breed 10 at a time
    ///
    /// # Parent Selection
    ///
    /// The `selector` determines how parents are chosen for breeding:
    /// - `Selector::tournament(3, 100)` - Tournament selection (size 3) from 100 candidates
    /// - `Selector::roulette(100)` - Fitness-proportionate selection from 100 candidates
    ///
    /// # Mutation Strategy
    ///
    /// The `mutagen` combines temperature (step size) and mutation rate (frequency):
    /// - `Mutagen::constant(0.5, 0.3)` - Fixed 50% temperature, 30% mutation rate
    /// - Adaptive strategies use `Temperature::linear()` and `MutationRate::exponential()`
    ///
    /// # Genetic Recombination
    ///
    /// The `crossover` method combines genetic material from parents:
    /// - `Crossover::uniform(0.5)` - Each gene has 50% chance from first parent
    /// - `Crossover::single_point()` - Cut genome at random point, swap tails
    ///
    /// # Initial Population
    ///
    /// The `distribution` strategy affects initial diversity and convergence:
    /// - `Distribution::latin_hypercube(50)` - Structured sampling for better coverage
    /// - `Distribution::random(50)` - Pure random sampling
    ///
    /// # Common Configurations
    ///
    /// ## Quick Convergence (Time-Constrained)
    /// ```rust,no_run
    /// use fx_durable_ga::models::*;
    /// # use fx_durable_ga::optimization::Service;
    /// # #[derive(Debug)] struct MyType;
    /// # impl Encodeable for MyType {
    /// #     const NAME: &'static str = "MyType";
    /// #     type Phenotype = MyType;
    /// #     fn morphology() -> Vec<GeneBounds> { vec![] }
    /// #     fn encode(&self) -> Vec<i64> { vec![] }
    /// #     fn decode(_: &[i64]) -> Self::Phenotype { MyType }
    /// # }
    /// # async fn example(service: &Service) -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// service.new_optimization_request(
    ///     MyType::NAME,
    ///     MyType::HASH,
    ///     FitnessGoal::maximize(0.90)?,                    // Modest target
    ///     Schedule::generational(20, 30),                 // Small, fast generations
    ///     Selector::tournament(5, 50),                    // Strong selection pressure
    ///     Mutagen::new(
    ///         Temperature::exponential(0.8, 0.1, 1.2, 3)?, // Rapid cooling
    ///         MutationRate::exponential(0.6, 0.05, 1.1, 2)?
    ///     ),
    ///     Crossover::uniform(0.6)?,                       // High recombination
    ///     Distribution::latin_hypercube(30)
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Balanced Search (General Purpose)
    /// ```rust,no_run
    /// use fx_durable_ga::models::*;
    /// # use fx_durable_ga::optimization::Service;
    /// # #[derive(Debug)] struct MyType;
    /// # impl Encodeable for MyType {
    /// #     const NAME: &'static str = "MyType";
    /// #     type Phenotype = MyType;
    /// #     fn morphology() -> Vec<GeneBounds> { vec![] }
    /// #     fn encode(&self) -> Vec<i64> { vec![] }
    /// #     fn decode(_: &[i64]) -> Self::Phenotype { MyType }
    /// # }
    /// # async fn example(service: &Service) -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// service.new_optimization_request(
    ///     MyType::NAME,
    ///     MyType::HASH,
    ///     FitnessGoal::maximize(0.95)?,                    // High-quality target
    ///     Schedule::generational(50, 50),                 // Moderate generations
    ///     Selector::tournament(3, 100),                   // Balanced selection
    ///     Mutagen::new(
    ///         Temperature::linear(0.7, 0.2, 1.0)?,         // Gradual cooling
    ///         MutationRate::linear(0.4, 0.1, 1.0)?
    ///     ),
    ///     Crossover::uniform(0.5)?,                       // Balanced recombination
    ///     Distribution::latin_hypercube(50)
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Thorough Exploration (Complex Problems)
    /// ```rust,no_run
    /// use fx_durable_ga::models::*;
    /// # use fx_durable_ga::optimization::Service;
    /// # #[derive(Debug)] struct MyType;
    /// # impl Encodeable for MyType {
    /// #     const NAME: &'static str = "MyType";
    /// #     type Phenotype = MyType;
    /// #     fn morphology() -> Vec<GeneBounds> { vec![] }
    /// #     fn encode(&self) -> Vec<i64> { vec![] }
    /// #     fn decode(_: &[i64]) -> Self::Phenotype { MyType }
    /// # }
    /// # async fn example(service: &Service) -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// service.new_optimization_request(
    ///     MyType::NAME,
    ///     MyType::HASH,
    ///     FitnessGoal::maximize(0.99)?,                    // High precision target
    ///     Schedule::rolling(10000, 200, 20),              // Large budget
    ///     Selector::tournament(2, 150),                   // Low selection pressure
    ///     Mutagen::constant(0.6, 0.4)?,                   // Sustained exploration
    ///     Crossover::single_point(),                      // Conservative recombination
    ///     Distribution::latin_hypercube(100)              // High initial diversity
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Parameter Interactions
    ///
    /// Key relationships to consider:
    /// - **High selection pressure + Low mutation**: Fast convergence, may get stuck
    /// - **Low selection pressure + High mutation**: Slow convergence, good exploration
    /// - **Large populations + Tournament selection**: Better genetic diversity
    /// - **Small populations + Roulette selection**: Risk of premature convergence
    /// - **Exponential decay + Generational schedule**: Classic simulated annealing
    /// - **Linear decay + Rolling schedule**: Steady refinement over time
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the optimization request was successfully created and scheduled.
    /// The actual optimization runs asynchronously via the event/job system.
    ///
    /// # Errors
    ///
    /// - Returns error if the specified type hasn't been registered with an evaluator
    /// - Returns error if database operations fail
    /// - Parameter validation errors are caught at construction time (e.g., `FitnessGoal::maximize(1.5)` fails)
    ///
    /// # Example Usage
    ///
    /// ```rust,no_run
    /// use fx_durable_ga::models::*;
    /// # use fx_durable_ga::optimization::Service;
    ///
    /// // Define your problem type
    /// #[derive(Debug)]
    /// struct Point { x: f64, y: f64 }
    ///
    /// impl Encodeable for Point {
    ///     const NAME: &'static str = "Point";
    ///     type Phenotype = Point;
    ///     fn morphology() -> Vec<GeneBounds> {
    ///         vec![
    ///             GeneBounds::decimal(0.0, 10.0, 1000, 3).unwrap(),
    ///             GeneBounds::decimal(0.0, 10.0, 1000, 3).unwrap(),
    ///         ]
    ///     }
    ///     fn encode(&self) -> Vec<i64> {
    ///         let bounds = Self::morphology();
    ///         vec![
    ///             bounds[0].from_sample(self.x / 10.0),
    ///             bounds[1].from_sample(self.y / 10.0),
    ///         ]
    ///     }
    ///     fn decode(genes: &[i64]) -> Self::Phenotype {
    ///         let bounds = Self::morphology();
    ///         Point {
    ///             x: bounds[0].decode_f64(genes[0]),
    ///             y: bounds[1].decode_f64(genes[1]),
    ///         }
    ///     }
    /// }
    ///
    /// # async fn example(service: &Service) -> Result<(), Box<dyn std::error::Error>> {
    /// // Start optimization
    /// service.new_optimization_request(
    ///     Point::NAME,
    ///     Point::HASH,
    ///     FitnessGoal::maximize(0.95)?,
    ///     Schedule::generational(50, 50),
    ///     Selector::tournament(3, 100),
    ///     Mutagen::constant(0.5, 0.3)?,
    ///     Crossover::uniform(0.5)?,
    ///     Distribution::latin_hypercube(50)
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "debug", skip(self), fields(type_name = type_name, type_hash = type_hash, goal = ?goal, mutagen = ?mutagen, crossover = ?crossover))]
    pub async fn new_optimization_request(
        &self,
        type_name: &str,
        type_hash: i32,
        goal: FitnessGoal,
        schedule: Schedule,
        selector: Selector,
        mutagen: Mutagen,
        crossover: Crossover,
        distribution: Distribution,
    ) -> Result<Uuid, Error> {
        tracing::info!("Optimization request received");

        let request = self
            .requests
            .chain(|mut tx_requests| {
                Box::pin(async move {
                    // Create a new optimization request with the repository
                    let request = tx_requests
                        .new_request(Request::new(
                            type_name,
                            type_hash,
                            goal,
                            selector,
                            schedule,
                            mutagen,
                            crossover,
                            distribution,
                        )?)
                        .await?;

                    // Instantiate a publisher
                    let mut publisher = fx_event_bus::Publisher::from_tx(tx_requests);

                    // Publish an event within the same transaction
                    publisher
                        .publish(OptimizationRequestedEvent::new(request.id))
                        .await?;

                    Ok((publisher, request))
                })
            })
            .await?;

        Ok(request.id)
    }

    /// Generates the initial population of genotypes for an optimization request.
    /// Creates genotypes based on the request's distribution and publishes generation events.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) async fn generate_initial_population(&self, request_id: Uuid) -> Result<(), Error> {
        tracing::info!("Generating initial population");

        // Get the optimization request
        let request = self.requests.get_request(request_id).await.map_err(|e| {
            tracing::error!(
                "Failed to fetch request in generate_initial_population with ID {}: {:?}",
                request_id,
                e
            );
            e
        })?;

        // Get the morphology of the type under optimization
        let morphology = self.morphologies.get_morphology(request.type_hash).await?;

        // Create an inital distribution of genomes
        let genomes = request.distribution.distribute(&morphology);

        let mut genotypes = Vec::with_capacity(genomes.len());
        let mut events = Vec::with_capacity(genomes.len());

        for genome in genomes {
            let genotype =
                Genotype::new(&request.type_name, request.type_hash, genome, request.id, 1);
            let event = GenotypeGenerated::new(request.id, genotype.id);

            genotypes.push(genotype);
            events.push(event);
        }

        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    // Create the initial generation atomically (prevents race conditions)
                    let inserted_genotypes = tx_genotypes.new_genotypes(genotypes).await?;

                    // Only proceed if genotypes were actually inserted (no race condition)
                    if !inserted_genotypes.is_empty() {
                        // Instantiate a publisher
                        let mut publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);

                        // Publish one event for each generated phenotype
                        publisher.publish_many(&events).await?;

                        Ok((publisher, request))
                    } else {
                        // Race condition occurred - another worker created the generation
                        // Create a dummy publisher to satisfy the return type
                        let publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);
                        Ok((publisher, request))
                    }
                })
            })
            .await?;

        Ok(())
    }

    /// Evaluates a genotype's fitness using the appropriate evaluator.
    /// Records the fitness result and publishes an evaluation event.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub(crate) async fn evaluate_genotype(
        &self,
        request_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        // Get the genotype from the database
        let genotype = self
            .genotypes
            .get_genotype(&genotype_id)
            .await
            .map_err(|e| {
                tracing::error!("Failed to fetch genotype with ID {}: {:?}", genotype_id, e);
                e
            })?;

        // Get the evaluator to use for this type
        let evaluator =
            self.evaluators
                .get(&genotype.type_hash)
                .ok_or(Error::UnknownTypeError {
                    type_hash: genotype.type_hash,
                    type_name: genotype.type_name,
                })?;

        // Call the evaluation function. This is a long running function!
        let terminator: Box<dyn Terminated> =
            Box::new(Terminator::new(self.requests.clone(), request_id));
        let fitness = evaluator.fitness(&genotype.genome, &terminator).await?;

        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    // Record the fitness of the genotype
                    tx_genotypes
                        .record_fitness(&Fitness::new(genotype_id, fitness))
                        .await?;

                    // Instantiate a publisher
                    let mut publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);

                    // Publish an event
                    publisher
                        .publish(GenotypeEvaluatedEvent::new(request_id, genotype_id))
                        .await?;

                    Ok((publisher, ()))
                })
            })
            .await?;

        Ok(())
    }

    #[instrument(level = "debug", skip(self, request), fields(request_id = %request.id, num_offspring = num_offspring, next_generation_id = next_generation_id))]
    async fn breed_genotypes(
        &self,
        request: &Request,
        num_offspring: usize,
        next_generation_id: i32,
    ) -> Result<(), Error> {
        // Early return if generation already exists (prevents duplicate work)
        let exists = self
            .genotypes
            .check_if_generation_exists(request.id, next_generation_id)
            .await?;
        if exists {
            return Ok(());
        }

        tracing::info!("Breeding genotypes");
        // Get candidates with fitness from populations repository
        let candidates_with_fitness = self
            .genotypes
            .search_genotypes(
                &genotypes::Filter::default()
                    .with_request_id(request.id)
                    .with_fitness(true)
                    .with_order_random(),
                request.selector.sample_size(),
            )
            .await?;

        // Get morphology for mutation bounds
        let morphology = self.morphologies.get_morphology(request.type_hash).await?;

        // Get current population to calculate optimization progress
        let population = self.genotypes.get_population(&request.id).await?;

        let best_fitness = *request
            .goal
            .best_fitness(&population.min_fitness, &population.max_fitness);

        // Iterative breeding with deduplication to avoid creating duplicate genomes
        let mut final_genotypes = Vec::with_capacity(num_offspring);
        let mut generated_hashes = HashSet::new();
        let mut deduplication_attempts = 0;

        while final_genotypes.len() < num_offspring
            && deduplication_attempts < self.max_deduplication_attempts
        {
            let needed = num_offspring - final_genotypes.len();

            // Re-select parents for this iteration (borrowed refs)
            let parent_pairs = request
                .selector
                .select_parents(needed, &candidates_with_fitness, &request.goal)
                .map_err(|e| Error::SelectionError(e))?;

            // Breed using Breeder (before async operations)
            let batch_genotypes = {
                let mut rng = rand::rng();
                Breeder::breed_batch(
                    &request,
                    &morphology,
                    &parent_pairs,
                    next_generation_id,
                    request.goal.calculate_progress(best_fitness),
                    &mut rng,
                )
            };

            // Collect hashes from this batch
            let batch_hashes: Vec<i64> = batch_genotypes.iter().map(|g| g.genome_hash).collect();

            // Check database for intersections
            let intersecting_hashes = self
                .genotypes
                .get_intersection(request.id, &batch_hashes)
                .await?
                .into_iter()
                .collect::<HashSet<i64>>();

            // Filter out duplicates (database + already generated)
            let mut unique_count = 0;
            for genotype in batch_genotypes {
                if !intersecting_hashes.contains(&genotype.genome_hash)
                    && !generated_hashes.contains(&genotype.genome_hash)
                {
                    generated_hashes.insert(genotype.genome_hash);
                    final_genotypes.push(genotype);
                    unique_count += 1;
                }
            }

            if unique_count < num_offspring {
                // Inform of duplicate generation. This is an indication that request params may require tuning.
                tracing::info!(
                    "Breeding generated {} non-unique genomes during attempt {}",
                    num_offspring - unique_count,
                    deduplication_attempts
                );
            }

            // Update deduplication_attempts counter
            if unique_count == 0 {
                deduplication_attempts += 1;
            } else {
                deduplication_attempts = 0;
            }
        }

        // Check if we exhausted max_deduplication_attempts progress iterations
        if deduplication_attempts >= self.max_deduplication_attempts {
            tracing::warn!(
                "Breeding exhausted max_deduplication_attempts ({}) for request {}. Consider tuning selection parameters or increasing diversity.",
                self.max_deduplication_attempts,
                request.id
            );
        }

        // If we couldn't generate enough unique genotypes, log a warning
        if final_genotypes.len() < num_offspring {
            tracing::warn!(
                "Could only generate {} unique genotypes out of {} requested for request {}",
                final_genotypes.len(),
                num_offspring,
                request.id
            );
        }

        // Update database (only if we have any genotypes)
        if !final_genotypes.is_empty() {
            // Create events for final genotypes
            let events: Vec<GenotypeGenerated> = final_genotypes
                .iter()
                .map(|genotype| GenotypeGenerated::new(request.id, genotype.id))
                .collect();

            self.genotypes
                .chain(|mut tx_genotypes| {
                    Box::pin(async move {
                        let inserted_genotypes =
                            tx_genotypes.new_genotypes(final_genotypes).await?;

                        // Only proceed if genotypes were actually inserted (no race condition)
                        if !inserted_genotypes.is_empty() {
                            let mut publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);
                            publisher.publish_many(&events).await?;

                            Ok((publisher, ()))
                        } else {
                            // Race condition occurred - another worker created the generation
                            let publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);
                            Ok((publisher, ()))
                        }
                    })
                })
                .await?;
        }

        Ok(())
    }

    /// Maintains the population by checking completion status and scheduling breeding or termination.
    /// Called after each genotype evaluation to determine the next optimization step.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) async fn maintain_population(&self, request_id: Uuid) -> Result<(), Error> {
        let key = format!("maintain_population_{}", request_id);
        self.locking
            .lock_while(&key, || async {
                // Check if request is already concluded - skip if so
                if let Some(_) = self.requests.get_request_conclusion(&request_id).await? {
                    tracing::debug!("Request already concluded, skipping maintenance");
                    return Ok(());
                }

                // Get the request
                let request = self.requests.get_request(request_id).await.map_err(|e| {
                    tracing::error!("Failed to fetch request with ID {}: {:?}", request_id, e);
                    e
                })?;

                // Get the population
                let population = self.genotypes.get_population(&request.id).await?;

                let Some(best_fitness) = *request
                    .goal
                    .best_fitness(&population.min_fitness, &population.max_fitness)
                else {
                    return Ok(());
                };

                if request.is_completed(best_fitness) {
                    // Publish RequestCompleted::new(request_id)
                    self.genotypes
                        .chain(|tx_genotypes| {
                            Box::pin(async move {
                                // Create publisher
                                let mut publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);

                                // Publish completion event
                                publisher
                                    .publish(RequestCompletedEvent::new(request.id))
                                    .await?;

                                Ok((publisher, ()))
                            })
                        })
                        .await?;
                    return Ok(());
                }

                match request.schedule.should_breed(&population) {
                    ScheduleDecision::Wait => Ok(()), // Do nothing
                    ScheduleDecision::Terminate => self.publish_terminated(request.id).await,
                    ScheduleDecision::Breed {
                        num_offspring,
                        next_generation_id,
                    } => {
                        self.breed_genotypes(&request, num_offspring, next_generation_id)
                            .await
                    }
                }
            })
            .await?
    }

    /// Records the conclusion of an optimization request to prevent duplicate processing.
    /// Uses locking to ensure atomicity when multiple workers might conclude the same request.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_conclusion.request_id, concluded_at = %request_conclusion.concluded_at, concluded_with = ?request_conclusion.concluded_with))]
    pub(crate) async fn conclude_request(
        &self,
        request_conclusion: RequestConclusion,
    ) -> Result<(), Error> {
        let key = format!("conclude_request_{}", request_conclusion.request_id);
        self.locking
            .lock_while(&key, || async {
                if let Some(_) = self
                    .requests
                    .get_request_conclusion(&request_conclusion.request_id)
                    .await?
                {
                    return Ok::<(), super::Error>(());
                }

                tracing::info!("Concluding request");

                self.requests
                    .new_request_conclusion(&request_conclusion)
                    .await?;

                Ok(())
            })
            .await?
    }

    /// Publishes a termination event for an optimization request.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) async fn publish_terminated(&self, request_id: Uuid) -> Result<(), Error> {
        self.genotypes
            .chain(|tx_genotypes| {
                Box::pin(async move {
                    let mut publisher = Publisher::new(tx_genotypes.tx());
                    let ret = publisher
                        .publish(RequestTerminatedEvent::new(request_id))
                        .await?;

                    Ok((publisher, ret))
                })
            })
            .await?;

        Ok(())
    }

    /// Gets the best genotype for a request based on fitness.
    /// Returns the genotype with the best (min or max) fitness depending on the goal.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn get_best_genotype(
        &self,
        request_id: Uuid,
    ) -> Result<Option<(Genotype, f64)>, Error> {
        let request = self.requests.get_request(request_id).await?;

        // Determine sort order based on goal
        let filter = match request.goal {
            FitnessGoal::Minimize { .. } => genotypes::Filter::default()
                .with_request_id(request_id)
                .with_fitness(true)
                .with_order_fitness_asc(),
            FitnessGoal::Maximize { .. } => genotypes::Filter::default()
                .with_request_id(request_id)
                .with_fitness(true)
                .with_order_fitness_desc(),
        };

        let results = self.genotypes.search_genotypes(&filter, 1).await?;

        // Extract the first result if it exists and has fitness
        let best = results
            .into_iter()
            .next()
            .and_then(|(genotype, fitness_opt)| fitness_opt.map(|fitness| (genotype, fitness)));

        Ok(best)
    }

    /// Checks if an optimization request has concluded.
    /// Returns true if the request has been completed or terminated.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn is_request_concluded(&self, request_id: Uuid) -> Result<bool, Error> {
        Ok(self.requests.get_request_conclusion(&request_id).await?.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap;
    use crate::models::{
        Crossover, Distribution, FitnessGoal, Mutagen, MutationRate, Schedule, Selector,
        Temperature,
    };

    async fn create_test_service(pool: sqlx::PgPool) -> Service {
        let builder = bootstrap::bootstrap(pool).await.unwrap();
        builder.build()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn test_new_optimization_request_happy_path(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Run event bus migrations for event publishing
        fx_event_bus::run_migrations(&pool).await?;

        // Create service
        let service = create_test_service(pool.clone()).await;

        // Create valid request parameters
        let type_name = "TestType";
        let type_hash = 123;
        let goal = FitnessGoal::maximize(0.95)?;
        let schedule = Schedule::generational(100, 10);
        let selector = Selector::tournament(3, 20);
        let mutagen = Mutagen::new(Temperature::constant(0.5)?, MutationRate::constant(0.3)?);
        let crossover = Crossover::uniform(0.5)?;
        let distribution = Distribution::random(50);

        // Call the method
        let result = service
            .new_optimization_request(
                type_name,
                type_hash,
                goal.clone(),
                schedule.clone(),
                selector.clone(),
                mutagen.clone(),
                crossover.clone(),
                distribution.clone(),
            )
            .await;

        // Verify success
        assert!(result.is_ok(), "new_optimization_request should succeed");

        // Verify request was stored in database
        let request_count = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM fx_durable_ga.requests WHERE type_name = $1 AND type_hash = $2",
            type_name,
            type_hash
        )
        .fetch_one(&pool)
        .await?
        .unwrap_or(0);

        assert_eq!(
            request_count, 1,
            "Exactly one request should be stored in database"
        );

        // Verify an OptimizationRequestedEvent was published
        let event_count: i64 = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM fx_event_bus.events_unacknowledged WHERE name = $1",
        )
        .bind("OptimizationRequested")
        .fetch_one(&pool)
        .await?;

        assert_eq!(
            event_count, 1,
            "Exactly one OptimizationRequestedEvent should be published"
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn test_generate_initial_population_happy_path(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Run event bus migrations
        fx_event_bus::run_migrations(&pool).await?;

        // Create service
        let service = create_test_service(pool.clone()).await;

        // Create a test morphology first
        let type_name = "TestType";
        let type_hash = 456;
        let morphology = crate::models::Morphology::new(
            type_name,
            type_hash,
            vec![
                crate::models::GeneBounds::integer(0, 9, 10).unwrap(), // 10 steps
                crate::models::GeneBounds::integer(0, 4, 5).unwrap(),  // 5 steps
            ],
        );

        // Insert morphology into database
        service.morphologies.new_morphology(morphology).await?;

        // Create a request
        let goal = FitnessGoal::maximize(0.95)?;
        let schedule = Schedule::generational(100, 10);
        let selector = Selector::tournament(3, 20);
        let mutagen = Mutagen::new(Temperature::constant(0.5)?, MutationRate::constant(0.3)?);
        let crossover = Crossover::uniform(0.5)?;
        let distribution = Distribution::random(10); // Small population for testing

        let request = crate::models::Request::new(
            type_name,
            type_hash,
            goal,
            selector,
            schedule,
            mutagen,
            crossover,
            distribution,
        )?;
        let request_id = request.id;

        // Insert request into database using chainable pattern
        service
            .requests
            .chain(|mut tx_requests| {
                Box::pin(async move {
                    let request = tx_requests.new_request(request).await?;
                    Ok((tx_requests, request))
                })
            })
            .await?;

        // Call the method under test
        let result = service.generate_initial_population(request_id).await;

        // Verify success
        assert!(result.is_ok(), "generate_initial_population should succeed");

        // Verify genotypes were created in database
        let genotype_count = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM fx_durable_ga.genotypes WHERE request_id = $1 AND generation_id = 1",
            request_id
        )
        .fetch_one(&pool)
        .await?
        .unwrap_or(0);

        assert_eq!(
            genotype_count, 10,
            "Exactly 10 genotypes should be created for generation 1"
        );

        // Verify GenotypeGenerated events were published
        let event_count: i64 = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM fx_event_bus.events_unacknowledged WHERE name = $1",
        )
        .bind("GenotypeGenerated")
        .fetch_one(&pool)
        .await?;

        assert_eq!(
            event_count, 10,
            "Exactly 10 GenotypeGenerated events should be published"
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn test_evaluate_genotype_happy_path(pool: sqlx::PgPool) -> anyhow::Result<()> {
        use crate::models::{Encodeable, Evaluator, GeneBounds, Terminated};
        use futures::future::BoxFuture;

        // Define a simple test type
        #[derive(Debug)]
        struct TestType;

        impl Encodeable for TestType {
            const NAME: &'static str = "TestType";
            type Phenotype = (i64, i64);

            fn morphology() -> Vec<GeneBounds> {
                vec![
                    GeneBounds::integer(0, 9, 10).unwrap(),
                    GeneBounds::integer(0, 4, 5).unwrap(),
                ]
            }

            fn encode(&self) -> Vec<i64> {
                vec![5, 2] // Not used in test
            }

            fn decode(genes: &[i64]) -> Self::Phenotype {
                (genes[0], genes[1])
            }
        }

        // Simple evaluator that returns constant fitness
        struct TestEvaluator;

        impl Evaluator<(i64, i64)> for TestEvaluator {
            fn fitness<'a>(
                &self,
                _phenotype: (i64, i64),
                _terminated: &'a Box<dyn Terminated>,
            ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
                Box::pin(async move { Ok(0.75) })
            }
        }

        // Run event bus migrations
        fx_event_bus::run_migrations(&pool).await?;

        // Create service and register evaluator
        let service = bootstrap::bootstrap(pool.clone())
            .await?
            .register::<TestType, _>(TestEvaluator)
            .await?
            .build();

        // Create a test request
        let goal = FitnessGoal::maximize(0.95)?;
        let schedule = Schedule::generational(100, 10);
        let selector = Selector::tournament(3, 20);
        let mutagen = Mutagen::new(Temperature::constant(0.5)?, MutationRate::constant(0.3)?);
        let crossover = Crossover::uniform(0.5)?;
        let distribution = Distribution::random(1);

        let request = crate::models::Request::new(
            TestType::NAME,
            TestType::HASH,
            goal,
            selector,
            schedule,
            mutagen,
            crossover,
            distribution,
        )?;
        let request_id = request.id;

        // Insert request into database
        service
            .requests
            .chain(|mut tx_requests| {
                Box::pin(async move {
                    let request = tx_requests.new_request(request).await?;
                    Ok((tx_requests, request))
                })
            })
            .await?;

        // Create a genotype manually
        let genotype = crate::models::Genotype::new(
            TestType::NAME,
            TestType::HASH,
            vec![7, 3], // Test genome
            request_id,
            1, // generation_id
        );
        let genotype_id = genotype.id;

        // Insert genotype into database
        service
            .genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    let genotypes = tx_genotypes.new_genotypes(vec![genotype]).await?;
                    Ok((tx_genotypes, genotypes))
                })
            })
            .await?;

        // Call the method under test
        let result = service.evaluate_genotype(request_id, genotype_id).await;

        // Verify success
        assert!(result.is_ok(), "evaluate_genotype should succeed");

        // Verify fitness was recorded in database
        let fitness = sqlx::query_scalar!(
            "SELECT fitness FROM fx_durable_ga.fitness WHERE genotype_id = $1",
            genotype_id
        )
        .fetch_one(&pool)
        .await?;

        assert_eq!(fitness, 0.75, "Fitness should be recorded as 0.75");

        // Verify GenotypeEvaluatedEvent was published
        let event_count: i64 = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM fx_event_bus.events_unacknowledged WHERE name = $1",
        )
        .bind("GenotypeEvaluated")
        .fetch_one(&pool)
        .await?;

        assert_eq!(
            event_count, 1,
            "Exactly one GenotypeEvaluatedEvent should be published"
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn test_maintain_population_request_completion(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Run event bus migrations
        fx_event_bus::run_migrations(&pool).await?;

        // Create service
        let service = create_test_service(pool.clone()).await;

        // Create morphology
        let type_name = "TestType";
        let type_hash = 789;
        let morphology = crate::models::Morphology::new(
            type_name,
            type_hash,
            vec![
                crate::models::GeneBounds::integer(0, 9, 10).unwrap(),
                crate::models::GeneBounds::integer(0, 4, 5).unwrap(),
            ],
        );
        service.morphologies.new_morphology(morphology).await?;

        // Create request with LOW fitness goal (0.5) so it's easy to exceed
        let goal = FitnessGoal::maximize(0.5)?;
        let schedule = Schedule::generational(100, 10);
        let selector = Selector::tournament(3, 20);
        let mutagen = Mutagen::new(Temperature::constant(0.5)?, MutationRate::constant(0.3)?);
        let crossover = Crossover::uniform(0.5)?;
        let distribution = Distribution::random(5);

        let request = crate::models::Request::new(
            type_name,
            type_hash,
            goal,
            selector,
            schedule,
            mutagen,
            crossover,
            distribution,
        )?;
        let request_id = request.id;

        // Insert request
        service
            .requests
            .chain(|mut tx_requests| {
                Box::pin(async move {
                    let request = tx_requests.new_request(request).await?;
                    Ok((tx_requests, request))
                })
            })
            .await?;

        // Create genotype with fitness that EXCEEDS the goal (0.8 > 0.5)
        let genotype =
            crate::models::Genotype::new(type_name, type_hash, vec![5, 2], request_id, 1);
        let genotype_id = genotype.id;

        // Insert genotype and fitness
        service
            .genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    // Insert genotype
                    tx_genotypes.new_genotypes(vec![genotype]).await?;

                    // Insert fitness that exceeds goal
                    let fitness = crate::models::Fitness::new(genotype_id, 0.8);
                    tx_genotypes.record_fitness(&fitness).await?;

                    Ok((tx_genotypes, ()))
                })
            })
            .await?;

        // Call the method under test
        let result = service.maintain_population(request_id).await;

        // Verify success
        assert!(result.is_ok(), "maintain_population should succeed");

        // Verify RequestCompletedEvent was published
        let event_count: i64 = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM fx_event_bus.events_unacknowledged WHERE name = $1",
        )
        .bind("RequestCompleted")
        .fetch_one(&pool)
        .await?;

        assert_eq!(
            event_count, 1,
            "Exactly one RequestCompletedEvent should be published"
        );

        Ok(())
    }
}
