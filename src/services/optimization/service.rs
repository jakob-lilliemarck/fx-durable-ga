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

// optimization service
pub struct Service {
    pub(super) locking: lock::Service,
    pub(super) requests: requests::Repository,
    pub(super) morphologies: morphologies::Repository,
    pub(super) genotypes: genotypes::Repository,
    pub(super) evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
    pub(super) max_deduplication_attempts: i32,
}

impl Service {
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

    #[instrument(level = "info", skip(self), fields(type_name = type_name, type_hash = type_hash, goal = ?goal, mutagen = ?mutagen, crossover = ?crossover))]
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
    ) -> Result<(), Error> {
        tracing::info!("Optimization request received");

        self.requests
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

        Ok(())
    }

    // Shall be run in a job triggered by OptimizationRequested
    #[instrument(level = "info", skip(self), fields(request_id = %request_id))]
    pub async fn generate_initial_population(&self, request_id: Uuid) -> Result<(), Error> {
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

    // Shall be run in a job triggerd by GenotypeGenerated
    #[instrument(level = "info", skip(self), fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub async fn evaluate_genotype(
        &self,
        request_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        tracing::info!("Evaluating genotype");

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

                // Get current population to calculate optimization progress
                let population = self.genotypes.get_population(&request.id).await?;

                // Iterative breeding with deduplication
                let mut final_genotypes = Vec::with_capacity(num_offspring);
                let mut generated_hashes = HashSet::new();
                let mut deduplication_attempts = 0;

                while final_genotypes.len() < num_offspring && deduplication_attempts < self.max_deduplication_attempts {
                    let needed = num_offspring - final_genotypes.len();

                    // Re-select parents for this iteration (borrowed refs)
                    let parent_pairs = request
                        .selector
                        .select_parents(needed, &candidates_with_fitness)
                        .map_err(|e| Error::SelectionError(e))?;

                    // Breed using Breeder (before async operations)
                    let batch_genotypes = {
                        let mut rng = rand::rng();
                        Breeder::breed_batch(
                            &request,
                            &morphology,
                            &parent_pairs,
                            next_generation_id,
                            request.goal.calculate_progress(population.best_fitness),
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
