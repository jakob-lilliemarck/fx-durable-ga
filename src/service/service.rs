use super::Error;
use super::models::{ErasedEvaluator, TypeErasedEvaluator};
use crate::models::{
    Crossover, Encodeable, Evaluator, Fitness, FitnessGoal, Genotype, Morphology, Mutagen, Request,
    Strategy,
};
use crate::repositories::chainable::{Chain, FromTx, ToTx};
use crate::repositories::{genotypes, morphologies, requests};
use crate::service::events::{
    GenotypeEvaluatedEvent, GenotypeGenerated, OptimizationRequestedEvent, RequestCompletedEvent,
    RequestTerminatedEvent,
};
use fx_event_bus::Publisher;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use tracing::instrument;
use uuid::Uuid;

// optimization service
pub struct Service {
    requests: requests::Repository,
    morphologies: morphologies::Repository,
    genotypes: genotypes::Repository,
    evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
}

pub struct ServiceBuilder {
    requests: requests::Repository,
    morphologies: morphologies::Repository,
    genotypes: genotypes::Repository,
    evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
}

impl ServiceBuilder {
    // NOTE:
    // This is a N+1, probably not so bad, it shouldn't be like this.
    // Its simple though, so I'll go with it for a first version
    #[instrument(level = "debug", skip(self, evaluator), fields(type_name = T::NAME, type_hash = T::HASH))]
    pub async fn register<T, E>(mut self, evaluator: E) -> Result<Self, Error>
    where
        T: Encodeable + 'static,
        E: Evaluator<T::Phenotype> + Send + Sync + 'static,
    {
        // Insert the morphology of the type if it does not already exist in the database
        if let Err(morphologies::Error::NotFound) = self.morphologies.get_morphology(T::HASH).await
        {
            self.morphologies
                .new_morphology(Morphology::new(T::NAME, T::HASH, T::morphology()))
                .await?;
        }

        // Erase the type
        let erased = ErasedEvaluator::new(evaluator, T::decode);

        // Insert it
        self.evaluators.insert(T::HASH, Box::new(erased));

        Ok(self)
    }

    #[instrument(level = "debug", skip(self), fields(evaluators_count = self.evaluators.len()))]
    pub fn build(self) -> Service {
        Service {
            requests: self.requests,
            morphologies: self.morphologies,
            genotypes: self.genotypes,
            evaluators: self.evaluators,
        }
    }
}

impl Service {
    pub(crate) fn builder(
        requests: requests::Repository,
        morphologies: morphologies::Repository,
        genotypes: genotypes::Repository,
    ) -> ServiceBuilder {
        ServiceBuilder {
            requests,
            morphologies,
            genotypes,
            evaluators: HashMap::new(),
        }
    }

    #[instrument(level = "info", skip(self), fields(type_name = type_name, type_hash = type_hash, goal = ?goal, temperature = temperature, mutation_rate = mutation_rate))]
    pub async fn new_optimization_request(
        &self,
        type_name: &str,
        type_hash: i32,
        goal: FitnessGoal,
        strategy: Strategy,
        temperature: f64,
        mutation_rate: f64,
    ) -> Result<(), Error> {
        let mutagen = Mutagen::constant(0.5, 0.1)?;
        let crossover = Crossover::uniform(0.5)?;

        self.requests
            .chain(|mut tx_requests| {
                Box::pin(async move {
                    // Create a new optimization request with the repository
                    let request = tx_requests
                        .new_request(Request::new(
                            type_name, type_hash, goal, strategy, mutagen, crossover,
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

        Ok(())
    }

    // Shall be run in a job triggered by OptimizationRequested
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn generate_initial_population(&self, request_id: Uuid) -> Result<(), Error> {
        // Get the optimization request
        tracing::debug!(
            "About to fetch request in generate_initial_population with ID: {}",
            request_id
        );
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

        // FIXME: DUPLICATE GENOMES PREVENTION
        // Currently using random genome generation which can create duplicate genotypes.
        // Should implement duplicate checking and regeneration to ensure genetic diversity.
        // Consider using HashSet<Vec<Gene>> to track generated genomes and retry on duplicates.

        let mut genotypes = Vec::with_capacity(request.population_size() as usize);
        let mut events = Vec::with_capacity(request.population_size() as usize);
        for _ in 0..request.population_size() {
            let genotype = Genotype::new(
                &request.type_name,
                request.type_hash,
                morphology.random(),
                request.id,
                1,
            );
            let event = GenotypeGenerated::new(request.id, genotype.id);

            genotypes.push(genotype);
            events.push(event);
        }

        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    // Create the initial generation atomically (prevents race conditions)
                    let inserted_genotypes = tx_genotypes
                        .create_generation_if_empty(request.id, 1, genotypes)
                        .await?;

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

    // Shall be run in a job triggerd by GenotypeGenerated
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub async fn evaluate_genotype(
        &self,
        request_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        // Get the genotype from the database
        tracing::debug!("About to fetch genotype with ID: {}", genotype_id);
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
        let fitness = evaluator.fitness(&genotype.genome).await?;

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

    #[instrument(level = "debug", skip(self, candidates_with_fitness), fields(num_pairs = num_pairs, tournament_size = tournament_size, candidates_count = candidates_with_fitness.len()))]
    fn select_by_tournament(
        &self,
        num_pairs: usize,
        tournament_size: usize,
        candidates_with_fitness: Vec<(Genotype, Option<f64>)>,
    ) -> Result<Vec<(Genotype, Genotype)>, Error> {
        let mut rng = rand::rng();
        let mut parent_pairs = Vec::with_capacity(num_pairs);

        // Filter out genotypes without fitness
        let evaluated_candidates: Vec<(Genotype, f64)> = candidates_with_fitness
            .into_iter()
            .filter_map(|(genotype, fitness_opt)| fitness_opt.map(|fitness| (genotype, fitness)))
            .collect();

        for _ in 0..num_pairs {
            let mut shuffled = evaluated_candidates.clone();
            shuffled.shuffle(&mut rng);

            // Tournament for parent 1
            let parent1 = shuffled[..tournament_size]
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .ok_or(Error::NoValidParents)?
                .0
                .clone();

            // Tournament for parent 2
            let parent2 = shuffled[tournament_size..(tournament_size * 2)]
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .ok_or(Error::NoValidParents)?
                .0
                .clone();

            parent_pairs.push((parent1, parent2));
        }

        Ok(parent_pairs)
    }

    #[instrument(level = "debug", skip(self, request), fields(request_id = %request.id, num_offspring = num_offspring, tournament_size = tournament_size, sample_size = sample_size, next_generation_id = next_generation_id))]
    async fn breed_new_genotypes(
        &self,
        request: &Request,
        num_offspring: usize,
        tournament_size: usize,
        sample_size: usize,
        next_generation_id: i32, // Same or incremented based on strategy
    ) -> Result<(), Error> {
        // Get candidates with fitness from populations repository
        let candidates_with_fitness = self
            .genotypes
            .search_genotypes(
                &genotypes::Filter::default()
                    .with_request_id(request.id)
                    .with_fitness(true)
                    .with_order_random(),
                sample_size as i64,
            )
            .await?;

        // Select parents - now returns Genotype pairs directly
        let parent_genotype_pairs =
            self.select_by_tournament(num_offspring, tournament_size, candidates_with_fitness)?;

        // Get morphology for mutation bounds
        let morphology = self.morphologies.get_morphology(request.type_hash).await?;

        // FIXME: DUPLICATE GENOMES PREVENTION
        // Crossover + mutation can produce duplicate genotypes, especially with low mutation rates.
        // Should implement duplicate checking against existing genotypes in the population.
        // Consider using database query or HashSet to detect and regenerate duplicates.

        // Create and mutate new offspring
        let mut new_genotypes = Vec::with_capacity(num_offspring);
        let mut events = Vec::with_capacity(num_offspring);
        for (a, b) in parent_genotype_pairs {
            let mut rng = rand::rng();

            let mut child = Genotype::new(
                &request.type_name,
                request.type_hash,
                request.crossover.apply(&mut rng, &a, &b),
                request.id,
                next_generation_id,
            );

            let mut rng = rand::rng();
            // FIXME:
            // 0.0 refers to the optimization progress. We must compute the progress. 0.0 represents full mutation strength, while 1.0 indicates zero mutation.
            // We shall provide a decreasing value as we progress towards an optimum - that should focus our efforts to the areas of interest that we have found
            request
                .mutagen
                .mutate(&mut rng, &mut child, &morphology, 0.0);

            events.push(GenotypeGenerated::new(request.id, child.id));
            new_genotypes.push(child);
        }

        // Update database
        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    let inserted_genotypes = tx_genotypes
                        .create_generation_if_empty(request.id, next_generation_id, new_genotypes)
                        .await?;

                    // Only proceed if genotypes were actually inserted (no race condition)
                    if !inserted_genotypes.is_empty() {
                        let mut publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);
                        publisher.publish_many(&events).await?;

                        Ok((publisher, ()))
                    } else {
                        // Race condition occurred - another worker created the generation
                        // Create a dummy publisher to satisfy the return type
                        let publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);
                        Ok((publisher, ()))
                    }
                })
            })
            .await?;

        Ok(())
    }

    // Shall be run in a job triggered by GenotypeEvaluated
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn maintain_population(&self, request_id: Uuid) -> Result<(), Error> {
        // Get the request
        let request = self.requests.get_request(request_id).await.map_err(|e| {
            tracing::error!("Failed to fetch request with ID {}: {:?}", request_id, e);
            e
        })?;

        // Get the population
        let population = self.genotypes.get_population(&request.id).await?;

        let best_fitness = match population.best_fitness {
            Some(fitness) => fitness,
            None => return Ok(()),
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

        match request.strategy {
            Strategy::Generational {
                max_generations,
                population_size,
            } => {
                // Guard: Max generations reached - terminate
                if max_generations <= population.current_generation as u32 {
                    self.publish_terminated(request.id).await?;
                    return Ok(());
                }

                // Guard: Still evaluating current generation - wait
                if population.live_genotypes > 0 {
                    return Ok(());
                }

                // Guard: Not enough evaluated genotypes to breed from - wait
                if population.evaluated_genotypes < population_size as i64 {
                    return Ok(());
                }

                // Happy path: breed new generation
                self.breed_new_genotypes(
                    &request,
                    population_size as usize,
                    (population_size / 8).max(2) as usize,
                    population_size as usize,
                    population.current_generation + 1,
                )
                .await?;
            }
            Strategy::Rolling {
                max_evaluations,
                population_size,
                selection_interval,
                tournament_size,
                sample_size,
            } => {
                // Guard: Haven't reached evaluation budget yet - wait
                if (max_evaluations as i64) > population.evaluated_genotypes {
                    return Ok(());
                }

                // Guard: Population full, can't breed anymore - terminate
                if population.live_genotypes >= (population_size - selection_interval) as i64 {
                    self.publish_terminated(request.id).await?;
                    return Ok(());
                }

                // Happy path: breed new genotypes
                self.breed_new_genotypes(
                    &request,
                    selection_interval as usize,
                    tournament_size as usize,
                    sample_size as usize,
                    population.current_generation + 1, // Increment generation for rolling strategies too
                )
                .await?;
            }
        }
        Ok(())
    }

    pub async fn publish_terminated(&self, request_id: Uuid) -> Result<(), Error> {
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
}
