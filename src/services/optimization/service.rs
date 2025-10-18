use super::Error;
use super::events::{
    GenotypeEvaluatedEvent, GenotypeGenerated, OptimizationRequestedEvent, RequestCompletedEvent,
    RequestTerminatedEvent,
};
use crate::models::{
    Crossover, Encodeable, Evaluator, Fitness, FitnessGoal, Genotype, Morphology, Mutagen, Request,
    RequestConclusion, Schedule, ScheduleDecision, Selector,
};
use crate::repositories::chainable::{Chain, FromTx, ToTx};
use crate::repositories::{genotypes, morphologies, requests};
use crate::services::lock;
use futures::future::BoxFuture;
use fx_event_bus::Publisher;
use std::collections::{HashMap, HashSet};
use tracing::instrument;
use uuid::Uuid;

// optimization service
pub struct Service {
    locking: lock::Service,
    requests: requests::Repository,
    morphologies: morphologies::Repository,
    genotypes: genotypes::Repository,
    evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
}

pub struct ServiceBuilder {
    locking: lock::Service,
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
            locking: self.locking,
            requests: self.requests,
            morphologies: self.morphologies,
            genotypes: self.genotypes,
            evaluators: self.evaluators,
        }
    }
}

impl Service {
    pub(crate) fn builder(
        locking: lock::Service,
        requests: requests::Repository,
        morphologies: morphologies::Repository,
        genotypes: genotypes::Repository,
    ) -> ServiceBuilder {
        ServiceBuilder {
            locking,
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
        tracing::info!("Optimization request received");

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
    #[instrument(level = "info", skip(self), fields(request_id = %request_id))]
    pub async fn generate_initial_population(&self, request_id: Uuid) -> Result<(), Error> {
        tracing::info!("Generating initial population");

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

    // Shall be run in a job triggerd by GenotypeGenerated
    #[instrument(level = "info", skip(self), fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub async fn evaluate_genotype(
        &self,
        request_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        tracing::info!("Evaluating genotype");

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

    /// Pure breeding logic - no database operations, returns offspring ready for deduplication
    fn breed_offspring_batch(
        request: &Request,
        parent_indices: Vec<(usize, usize)>,
        candidates_with_fitness: &[(Genotype, Option<f64>)],
        morphology: &Morphology,
        next_generation_id: i32,
    ) -> Vec<Genotype> {
        let mut new_genotypes = Vec::with_capacity(parent_indices.len());

        for (idx1, idx2) in parent_indices {
            // Get parents by reference to avoid cloning (indices should be valid from selector)
            let parent1 = &candidates_with_fitness[idx1].0;
            let parent2 = &candidates_with_fitness[idx2].0;

            let mut rng = rand::rng();

            let mut child = Genotype::new(
                &request.type_name,
                request.type_hash,
                request.crossover.apply(&mut rng, parent1, parent2),
                request.id,
                next_generation_id,
            );

            let mut rng = rand::rng();
            // FIXME: 0.0 refers to optimization progress - should be computed
            request
                .mutagen
                .mutate(&mut rng, &mut child, morphology, 0.0);

            new_genotypes.push(child);
        }

        new_genotypes
    }

    #[instrument(level = "info", skip(self, request), fields(request_id = %request.id, num_offspring = num_offspring, next_generation_id = next_generation_id))]
    async fn breed_genotypes(
        &self,
        request: &Request,
        num_offspring: usize,
        next_generation_id: i32,
    ) -> Result<(), Error> {
        let key = format!("request_{}", request.id);
        let ret = self
            .locking
            .lock_while(&key, || async {
                // Early return if it exists!
                let exists = self.genotypes.check_if_generation_exists(request.id, next_generation_id).await?;
                if exists {
                    return Ok(())
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

                // Iterative breeding with deduplication
                let mut final_genotypes = Vec::with_capacity(num_offspring);
                let mut generated_hashes = HashSet::new();
                let mut zero_progress_counter = 0;
                const MAX_ZERO_PROGRESS: i32 = 5;

                while final_genotypes.len() < num_offspring && zero_progress_counter < MAX_ZERO_PROGRESS {
                    let needed = num_offspring - final_genotypes.len();

                    // Re-select parents for this iteration
                    let parent_indices = request
                        .selector
                        .select_parents(needed, candidates_with_fitness.clone())
                        .map_err(|e| Error::SelectionError(e))?;

                    // Breed a batch directly from indices
                    let batch_genotypes = Self::breed_offspring_batch(
                        request,
                        parent_indices,
                        &candidates_with_fitness,
                        &morphology,
                        next_generation_id,
                    );

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

                    // Update zero progress counter
                    if unique_count == 0 {
                        zero_progress_counter += 1;
                    } else {
                        zero_progress_counter = 0;
                    }
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

                    // FIXME:
                    // It could be that the lock acquired inside insert_generation is released BEFORE the transaction is commited.
                    // if that is the case, it means other queries will start to run then, not yet seeing the work carried out by THIS transaction.
                    // Also we do unneccesary work if we discard works - we might as well open a lock at the beginning of this whole function call, and hold it to the end of the call?
                    self.genotypes
                        .chain(|mut tx_genotypes| {
                            Box::pin(async move {
                                let inserted_genotypes = tx_genotypes
                                    .new_genotypes(final_genotypes)
                                    .await?;

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
            }).await?;
        ret
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
                self.breed_genotypes(&request, num_offspring, next_generation_id)
                    .await
            }
        }
    }

    #[instrument(level = "info", skip(self), fields(request_id = %request_conclusion.request_id, concluded_at = %request_conclusion.concluded_at, concluded_with = ?request_conclusion.concluded_with))]
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

pub(crate) trait TypeErasedEvaluator: Send + Sync {
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}

pub(crate) struct ErasedEvaluator<P, E: Evaluator<P>> {
    evaluator: E,
    decode: fn(&[i64]) -> P,
}

impl<P, E: Evaluator<P>> ErasedEvaluator<P, E> {
    pub(crate) fn new(evaluator: E, decode: fn(&[i64]) -> P) -> Self {
        Self { evaluator, decode }
    }
}

impl<P, E> TypeErasedEvaluator for ErasedEvaluator<P, E>
where
    E: Evaluator<P> + Send + Sync + 'static,
{
    #[instrument(level = "debug", skip(self, genes), fields(genome_length = genes.len()))]
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
        let phenotype = (self.decode)(genes);
        self.evaluator.fitness(phenotype)
    }
}
