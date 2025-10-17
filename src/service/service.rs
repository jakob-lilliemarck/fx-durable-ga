use super::Error;
use super::models::{ErasedEvaluator, TypeErasedEvaluator};
use crate::models::{
    Crossover, Encodeable, Evaluator, Fitness, FitnessGoal, Genotype, Morphology, Mutagen, Request,
    Schedule, ScheduleDecision, Selector,
};
use crate::repositories::chainable::{Chain, FromTx, ToTx};
use crate::repositories::{genotypes, morphologies, requests};
use crate::service::events::{
    GenotypeEvaluatedEvent, GenotypeGenerated, OptimizationRequestedEvent, RequestCompletedEvent,
    RequestTerminatedEvent,
};
use fx_event_bus::Publisher;
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
        schedule: Schedule,
        selector: Selector,
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
                            type_name, type_hash, goal, selector, schedule, mutagen, crossover,
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

    #[instrument(level = "debug", skip(self, request), fields(request_id = %request.id, num_offspring = num_offspring, next_generation_id = next_generation_id))]
    async fn breed_new_genotypes(
        &self,
        request: &Request,
        num_offspring: usize,
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
                request.selector.sample_size(),
            )
            .await?;

        // Use the new Selector to get parent indices
        let parent_indices = request
            .selector
            .select_parents(num_offspring, candidates_with_fitness.clone())
            .map_err(|e| Error::SelectionError(e))?; // You'll need to add this to your Error enum

        // Convert indices to actual genotypes
        let parent_genotype_pairs = parent_indices
            .into_iter()
            .map(|(idx1, idx2)| {
                let parent1 = candidates_with_fitness
                    .get(idx1)
                    .ok_or(Error::InvalidParentIndex(idx1))?
                    .0
                    .clone();
                let parent2 = candidates_with_fitness
                    .get(idx2)
                    .ok_or(Error::InvalidParentIndex(idx2))?
                    .0
                    .clone();
                Ok((parent1, parent2))
            })
            .collect::<Result<Vec<(Genotype, Genotype)>, Error>>()?;

        // Get morphology for mutation bounds
        let morphology = self.morphologies.get_morphology(request.type_hash).await?;

        // Create and mutate new offspring
        let mut new_genotypes = Vec::with_capacity(num_offspring);
        let mut events = Vec::with_capacity(num_offspring);

        for (parent1, parent2) in parent_genotype_pairs {
            let mut rng = rand::rng();

            let mut child = Genotype::new(
                &request.type_name,
                request.type_hash,
                request.crossover.apply(&mut rng, &parent1, &parent2),
                request.id,
                next_generation_id,
            );

            let mut rng = rand::rng();
            // FIXME: 0.0 refers to optimization progress - should be computed
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

        match request.schedule.should_breed(&population) {
            ScheduleDecision::Wait => Ok(()), // Do nothing
            ScheduleDecision::Terminate => self.publish_terminated(request.id).await,
            ScheduleDecision::Breed {
                num_offspring,
                next_generation_id,
            } => {
                self.breed_new_genotypes(&request, num_offspring, next_generation_id)
                    .await
            }
        }
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
