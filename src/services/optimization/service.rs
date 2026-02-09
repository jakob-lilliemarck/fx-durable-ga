use super::Error;
use super::events::{
    GenotypeEvaluatedEvent, GenotypeGeneratedEvent, OptimizationRequestedEvent,
    RequestCompletedEvent, RequestInterruptedEvent, RequestTerminatedEvent,
};
use crate::models::GenotypeManager;
use crate::models::{
    Breeder, Evaluation, FitnessGoal, Genotype, Request, RequestConclusion, ScheduleDecision,
    Selector,
};
use crate::repositories::chainable::{Chain, FromTx, ToTx};
use crate::repositories::{genotypes, requests};
use crate::services::lock;
use crate::services::optimization::termination_listener::TerminationListener;
use chrono::Utc;
use fx_event_bus::Publisher;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

/// Genetic algorithm optimization service that manages the entire optimization lifecycle.
pub struct Service {
    pub(super) host_id: Uuid,
    pub(super) locking: Arc<lock::Service>,
    pub(super) requests: Arc<requests::Repository>,
    pub(super) genotypes: Arc<genotypes::Repository>,
    pub(super) genotype_managers: HashMap<i32, Box<dyn GenotypeManager + 'static>>,
    pub(super) max_deduplication_attempts: i32,
    pub(super) termination_listener: TerminationListener,
}

impl Service {
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn builder(
        host_id: Uuid,
        locking: &Arc<lock::Service>,
        requests: &Arc<requests::Repository>,
        genotypes: &Arc<genotypes::Repository>,
    ) -> super::ServiceBuilder {
        super::ServiceBuilder {
            host_id,
            locking: locking.clone(),
            requests: requests.clone(),
            genotypes: genotypes.clone(),
            genotype_managers: HashMap::new(),
            max_deduplication_attempts: 5,
        }
    }

    #[instrument(level = "debug", skip(self, data), fields(type_name = type_name, type_hash = type_hash, goal = ?goal))]
    pub async fn new_optimization_request(
        &self,
        type_name: &str,
        type_hash: i32,
        goal: FitnessGoal,
        schedule: crate::models::Schedule,
        selector: Selector,
        user_defined: impl serde::Serialize + Send + Sync + std::fmt::Debug,
        data: Option<impl serde::Serialize + Send + Sync>,
    ) -> Result<Uuid, Error> {
        if !self.genotype_managers.contains_key(&type_hash) {
            return Err(Error::UnknownTypeError {
                type_hash,
                type_name: type_name.to_string(),
            });
        }

        let request = self
            .requests
            .chain(|mut tx_requests| {
                Box::pin(async move {
                    let request = tx_requests
                        .new_request(Request::new(
                            type_name,
                            type_hash,
                            goal,
                            selector,
                            schedule,
                            user_defined,
                            data,
                        )?)
                        .await?;

                    let mut publisher = fx_event_bus::Publisher::from_tx(tx_requests);
                    publisher
                        .publish(OptimizationRequestedEvent::new(request.id))
                        .await?;

                    Ok((publisher, request))
                })
            })
            .await?;

        Ok(request.id)
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) async fn generate_initial_population(&self, request_id: Uuid) -> Result<(), Error> {
        let request = self.requests.get_request(request_id).await?;
        let manager =
            self.genotype_managers
                .get(&request.type_hash)
                .ok_or(Error::UnknownTypeError {
                    type_hash: request.type_hash,
                    type_name: request.type_name.clone(),
                })?;

        let population_size = request.schedule.population_size as usize;
        let mut genotypes = Vec::with_capacity(population_size);
        let mut events = Vec::with_capacity(population_size);
        {
            let mut rng = rand::rng();
            for _ in 0..population_size {
                let genome = manager
                    .random(&mut rng, &request.user_defined)
                    .map_err(Error::EvaluationError)?;
                let genotype = Genotype::new(
                    &request.type_name,
                    request.type_hash,
                    genome,
                    request.id,
                    1,    // First generation
                    None, // No parent_a
                    None, // No parent_b
                );
                events.push(GenotypeGeneratedEvent::new(request.id, genotype.id()));
                genotypes.push(genotype);
            }
        }

        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    let inserted = tx_genotypes.new_genotypes(genotypes).await?;
                    if !inserted.is_empty() {
                        let mut publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);
                        publisher.publish_many(&events).await?;
                        Ok((publisher, ()))
                    } else {
                        let publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);
                        Ok((publisher, ()))
                    }
                })
            })
            .await?;

        Ok(())
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub(crate) async fn evaluate_genotype(
        &self,
        request_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        let genotype = self.genotypes.get_genotype(&genotype_id).await?;
        let request = self.requests.get_request(request_id).await?;

        let manager =
            self.genotype_managers
                .get(&genotype.type_hash())
                .ok_or(Error::UnknownTypeError {
                    type_hash: genotype.type_hash(),
                    type_name: genotype.type_name().to_string(),
                })?;

        let genome = genotype.genome();
        let started_at = Utc::now();

        // Race evaluation future against termination notifications so we can stop
        // once a request concludes.
        let (fitness, started_at, completed_at) = tokio::select! {
            res = manager.evaluate(&genome, &request.user_defined) => {


                // Map error and escape early
                let fitness = res.map_err(Error::EvaluationError)?;

                // Evaluation completed at this time
                let completed_at = Utc::now();

                (fitness, started_at, completed_at)
            }
            termination = self.termination_listener.wait_for(request_id) => {
                termination?;
                return Ok(());
            }
        };

        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    tx_genotypes
                        .record_evaluation(&Evaluation::new(
                            genotype_id,
                            fitness,
                            Some(started_at),
                            Some(completed_at),
                            Some(self.host_id.clone()),
                        ))
                        .await?;
                    let mut publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);
                    publisher
                        .publish(GenotypeEvaluatedEvent::new(
                            request_id,
                            genotype.generation_id(),
                            genotype_id,
                        ))
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
        // Only breed if there is no current generation with this ID
        if self
            .genotypes
            .check_if_generation_exists(request.id, next_generation_id)
            .await?
        {
            tracing::warn!(
                "breed_genotypes was called for a generation that already exists. Generation ID: {}",
                next_generation_id
            );
            return Ok(());
        }

        // Get the GenotypeManager from the service
        let manager =
            self.genotype_managers
                .get(&request.type_hash)
                .ok_or(Error::UnknownTypeError {
                    type_hash: request.type_hash,
                    type_name: request.type_name.clone(),
                })?;

        // Start building up the filters for retrieving a sample of genotypes to select parents from
        let mut selection_filter = genotypes::GenotypesFilter::default()
            .with_request_id(request.id)
            .with_evaluation(true);

        if request.schedule.is_generational() {
            // If we're using a generational schedule, then make sure to only query the last generation of this request
            selection_filter = selection_filter
                .with_generation_id(next_generation_id - 1)
                .with_order_random();
        } else {
            // Otherwise just get a window of N last evaluations
            selection_filter = selection_filter.with_order_completed_at_desc();
        }

        // Get candidates with fitness using the filter.
        // Error out if any candidate is missing fitness
        let candidates_with_fitness = self
            .genotypes
            .search_genotypes(&selection_filter, request.schedule.population_size() as i64)
            .await?
            .into_iter()
            .map(|(genotype, fitness_opt)| match fitness_opt {
                Some(fitness) => Ok((genotype, fitness)),
                None => Err(Error::NoFitness {
                    genotype_id: genotype.id,
                }),
            })
            .collect::<Result<Vec<(Genotype, f64)>, Error>>()?;

        // Pass candidates with fitness to the selector to get pairs of selected parents
        let pairs = request.selector.select_parents(
            num_offspring,
            &candidates_with_fitness,
            &request.goal,
        )?;

        // Get a set of children enumerated by their parent pair index
        let (batch, child_hashes) = Breeder::breed_batch(
            request,
            manager.as_ref(),
            &pairs,
            next_generation_id,
            self.max_deduplication_attempts as usize,
        )?;

        // Intersect the hashes with the database
        let intersection = self
            .genotypes
            .get_intersection(request.id, &child_hashes)
            .await?;

        let existing: HashSet<i64> = intersection.keys().copied().collect();
        let deduplicated = Breeder::deduplicate_batch(existing, batch);

        // Evaluations that could be written from cached fitness
        let mut evaluations: Vec<Evaluation> = Vec::new();
        // Genotypes
        let mut genotypes: Vec<Genotype> = Vec::with_capacity(num_offspring);
        // GenotypeGenerated events for each genotype not present in the database
        let mut events: Vec<GenotypeGeneratedEvent> = Vec::with_capacity(num_offspring);

        for d in deduplicated {
            if d.existing {
                // The hash must be in intersection
                let (_, e) = intersection.get(&d.genotype.genome_hash).unwrap();
                // Write evaluations for already evaluated genomes using their fitness
                evaluations.push(Evaluation::new_with_copied_from(
                    d.genotype.id(),
                    e.fitness,
                    Utc::now(),
                    Utc::now(),
                    self.host_id,
                    *e.genotype_id(),
                ))
            } else {
                events.push(GenotypeGeneratedEvent::new(request.id, d.genotype.id()))
            }
            // Always push all genotypes
            genotypes.push(d.genotype);
        }

        self.genotypes
            .chain(|mut tx| {
                Box::pin(async move {
                    // actually we should insert new AND ext Genotypes
                    let inserted = tx.new_genotypes(genotypes).await?;

                    if !evaluations.is_empty() {
                        tx.record_evaluations(&evaluations).await?;
                    }

                    let mut publisher = fx_event_bus::Publisher::from_tx(tx);
                    if !events.is_empty() && !inserted.is_empty() {
                        publisher.publish_many(&events).await?;
                    }

                    Ok((publisher, ()))
                })
            })
            .await?;

        Ok(())
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) async fn maintain_population(&self, request_id: Uuid) -> Result<(), Error> {
        let key = format!("maintain_population_{}", request_id);
        self.locking
            .lock_while(&key, || async {
                if let Some(_) = self.requests.get_request_conclusion(&request_id).await? {
                    return Ok(());
                }

                let request = self.requests.get_request(request_id).await?;
                let population = self.genotypes.get_population(&request.id).await?;

                let Some(best_fitness) = *request
                    .goal
                    .best_fitness(&population.min_fitness, &population.max_fitness)
                else {
                    return Ok(());
                };

                if request.is_completed(best_fitness) {
                    self.genotypes
                        .chain(|tx_genotypes| {
                            Box::pin(async move {
                                let mut publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);
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
                    ScheduleDecision::Wait => Ok(()),
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

    #[instrument(level = "debug", skip(self), fields(request_id = %request_conclusion.request_id, concluded_at = %request_conclusion.concluded_at, concluded_with = ?request_conclusion.concluded_with))]
    pub(crate) async fn conclude_request(
        &self,
        request_conclusion: RequestConclusion,
    ) -> Result<(), Error> {
        let request_id = request_conclusion.request_id;
        let key = format!("conclude_request_{}", request_id);
        let _ = self
            .locking
            .lock_while(&key, || async {
                if let Some(_) = self
                    .requests
                    .get_request_conclusion(&request_conclusion.request_id)
                    .await?
                {
                    return Ok::<(), super::Error>(());
                }

                self.requests
                    .new_request_conclusion(&request_conclusion)
                    .await?;

                self.requests.notify_request_conclusion(&request_id).await?;

                Ok(())
            })
            .await?;

        Ok(())
    }

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

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn interrupt_request(&self, request_id: Uuid) -> Result<(), Error> {
        self.genotypes
            .chain(|tx_genotypes| {
                Box::pin(async move {
                    let mut publisher = Publisher::new(tx_genotypes.tx());
                    let ret = publisher
                        .publish(RequestInterruptedEvent::new(request_id))
                        .await?;

                    Ok((publisher, ret))
                })
            })
            .await?;

        Ok(())
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn get_best_genotype(
        &self,
        request_id: Uuid,
    ) -> Result<Option<(Genotype, f64)>, Error> {
        let request = self.requests.get_request(request_id).await?;

        let filter = match request.goal {
            FitnessGoal::Minimize { .. } => genotypes::GenotypesFilter::default()
                .with_request_id(request_id)
                .with_evaluation(true)
                .with_order_fitness_asc(),
            FitnessGoal::Maximize { .. } => genotypes::GenotypesFilter::default()
                .with_request_id(request_id)
                .with_evaluation(true)
                .with_order_fitness_desc(),
        };

        let results = self.genotypes.search_genotypes(&filter, 1).await?;
        let best = results
            .into_iter()
            .next()
            .and_then(|(genotype, fitness_opt)| fitness_opt.map(|fitness| (genotype, fitness)));

        Ok(best)
    }

    #[instrument(level = "debug", skip(self), fields(type_name))]
    pub async fn search_genotypes(
        &self,
        filter: &genotypes::GenotypesFilter,
        limit: i64,
    ) -> Result<Vec<(Genotype, Option<f64>)>, Error> {
        let genotypes = self.genotypes.search_genotypes(filter, limit).await?;
        Ok(genotypes)
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn is_request_concluded(&self, request_id: Uuid) -> Result<bool, Error> {
        Ok(self
            .requests
            .get_request_conclusion(&request_id)
            .await?
            .is_some())
    }
}
