use crate::repositories::chainable::{Chain, FromOther, FromTx};
use crate::repositories::genotypes::Genotype;
use crate::repositories::populations;
use crate::repositories::{
    genotypes,
    morphologies::{self, Morphology},
    requests::{self, Request},
};
use crate::service::events::{
    GenotypeEvaluatedEvent, GenotypeGenerated, OptimizationRequestedEvent, RequestCompletedEvent,
    RequestTerminatedEvent,
};
use const_fnv1a_hash::fnv1a_hash_str_32;
use futures::future::BoxFuture;
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("RequestsRepositoryError: {0}")]
    RequestsRepositoryError(#[from] requests::Error),
    #[error("MorphologiesRepositoryError: {0}")]
    MorphologiesRepositoryError(#[from] morphologies::Error),
    #[error("GenotypesRepositoryError: {0}")]
    GenotypesRepositoryError(#[from] genotypes::Error),
    #[error("PopulationsRepositoryError: {0}")]
    PopulationsRepositoryError(#[from] populations::Error),
    #[error("UnknownType: type_name={type_name}, type_hash={type_hash}")]
    UnknownTypeError { type_hash: i32, type_name: String },
    #[error("EvaluationError: {0}")]
    EvaluationError(#[from] anyhow::Error),
    #[error("NoValidParents")]
    NoValidParents,
    #[error("PublishErrorTemp")]
    PublishErrorTemp, //FIXME
}

pub trait Encodeable {
    const NAME: &str;
    const HASH: i32 = fnv1a_hash_str_32(Self::NAME) as i32;

    type Phenotype;

    fn morphology() -> Vec<morphologies::GeneBounds>;
    fn encode(&self) -> Vec<i64>;
    fn decode(genes: &[i64]) -> Self::Phenotype;
}

pub trait Evaluator<P> {
    fn fitness<'a>(&self, phenotype: P) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}

trait TypeErasedEvaluator: Send + Sync {
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}

struct ErasedEvaluator<P, E: Evaluator<P>> {
    evaluator: E,
    decode: fn(&[i64]) -> P,
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

// optimization service
pub struct Service {
    requests: requests::Repository,
    morphologies: morphologies::Repository,
    genotypes: genotypes::Repository,
    populations: populations::Repository,
    evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
}

pub struct ServiceBuilder {
    requests: requests::Repository,
    morphologies: morphologies::Repository,
    genotypes: genotypes::Repository,
    populations: populations::Repository,
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
        let erased = ErasedEvaluator {
            evaluator,
            decode: T::decode,
        };

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
            populations: self.populations,
            evaluators: self.evaluators,
        }
    }
}

impl Service {
    pub fn builder(
        requests: requests::Repository,
        morphologies: morphologies::Repository,
        genotypes: genotypes::Repository,
        populations: populations::Repository,
    ) -> ServiceBuilder {
        ServiceBuilder {
            requests,
            morphologies,
            genotypes,
            populations,
            evaluators: HashMap::new(),
        }
    }

    #[instrument(level = "info", skip(self), fields(type_name = type_name, type_hash = type_hash, goal = ?goal, threshold = threshold, temperature = temperature, mutation_rate = mutation_rate))]
    pub async fn new_optimization_request(
        &self,
        type_name: &str,
        type_hash: i32,
        goal: requests::FitnessGoal,
        threshold: f64,
        strategy: requests::Strategy,
        temperature: f64,
        mutation_rate: f64,
    ) -> Result<(), Error> {
        self.requests
            .chain(|mut tx_requests| {
                Box::pin(async move {
                    // Create a new optimization request with the repository
                    let request = tx_requests
                        .new_request(Request::new(
                            type_name,
                            type_hash,
                            goal,
                            threshold,
                            strategy,
                            temperature,
                            mutation_rate,
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

        let mut genotypes = Vec::with_capacity(request.population_size() as usize);
        let mut population = Vec::with_capacity(request.population_size() as usize);
        let mut events = Vec::with_capacity(request.population_size() as usize);
        for _ in 0..request.population_size() {
            let genotype = genotypes::Genotype::new(
                &request.type_name,
                request.type_hash,
                morphology.random(),
                request.id,
                1,
            );
            population.push((request.id, genotype.id));
            let event = GenotypeGenerated::new(request.id, genotype.id);

            genotypes.push(genotype);
            events.push(event);
        }

        let populations = self.populations.clone();
        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    // Create the genotypes with the repository
                    tx_genotypes.new_genotypes(genotypes).await?;

                    let mut tx_populations = populations.from_other(tx_genotypes);

                    // Add all to the active population of this request
                    tx_populations.add_to_population(&population).await?;

                    // Instantiate a publisher
                    let mut publisher = fx_event_bus::Publisher::from_tx(tx_populations);

                    // Publish one event for each generated phenotype
                    publisher.publish_many(&events).await?;

                    Ok((publisher, request))
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

        let populations = self.populations.clone();
        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    // Update the fitness of the genotype
                    tx_genotypes.set_fitness(genotype_id, fitness).await?;

                    let mut tx_populations = populations.from_other(tx_genotypes);

                    // Remove the genotype from the activde population
                    tx_populations
                        .remove_from_population(&request_id, &genotype_id)
                        .await?;

                    // Instantiate a publisher
                    let mut publisher = fx_event_bus::Publisher::from_tx(tx_populations);

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

    /// Creates child genome by mixing genes from two parents
    #[instrument(level = "debug", skip(self, a, b), fields(parent_a_id = %a.id, parent_b_id = %b.id, request_id = %request_id, generation_id = generation_id))]
    fn crossover(
        &self,
        a: &Genotype,
        b: &Genotype,
        request_id: Uuid,
        generation_id: i32,
    ) -> Genotype {
        let mut rng = rand::rng();
        // Option 1: Simple uniform crossover (50/50 chance for each gene)
        let genome: Vec<genotypes::Gene> = a
            .genome
            .iter()
            .zip(b.genome.iter())
            .map(|(&a, &b)| if rng.random_bool(0.5) { a } else { b })
            .collect();

        Genotype::new(&a.type_name, a.type_hash, genome, request_id, generation_id)
    }

    // NOTE: temperatur and rate should have been validated at the time of creating the request, we do not validate again at this point.
    #[instrument(level = "debug", skip(self, genotype, morphology), fields(genotype_id = %genotype.id, temperature = temperature, rate = rate))]
    fn mutate(
        &self,
        genotype: &mut Genotype,
        morphology: &Morphology,
        temperature: f64,
        rate: f64,
    ) {
        let mut rng = rand::rng();
        for (gene, bounds) in genotype
            .genome
            .iter_mut()
            .zip(morphology.gene_bounds.iter())
        {
            // Should we mutate this gene?
            if rng.random_range(0.0..1.0) < rate {
                // Temperature controls mutation step: higher = larger jumps
                let max_step = (1.0 + (bounds.divisor as f64 * temperature)) as i64;

                // Choose direction and step size
                let direction = if rng.random_bool(0.5) { 1 } else { -1 };
                let step = rng.random_range(1..=max_step);

                // Apply mutation and clamp
                *gene = (*gene + direction * step).clamp(0, bounds.divisor as i64 - 1);
            }
        }
    }

    #[instrument(level = "debug", skip(self, candidates), fields(num_pairs = num_pairs, tournament_size = tournament_size, candidates_count = candidates.len()))]
    fn select_by_tournament(
        &self,
        num_pairs: usize,
        tournament_size: usize,
        mut candidates: Vec<Genotype>,
    ) -> Result<Vec<(Genotype, Genotype)>, Error> {
        let mut rng = rand::rng();
        let mut pairs = Vec::with_capacity(num_pairs);

        for _ in 0..num_pairs {
            // Shuffle once at the start of each iteration
            candidates.shuffle(&mut rng);

            // Take first tournament_size elements for first parent
            // Panics if fitness is None - but it is never expected to be
            let parent1 = candidates[..tournament_size]
                .iter()
                .max_by(|a, b| a.must_fitness().partial_cmp(&b.must_fitness()).unwrap())
                .ok_or(Error::NoValidParents)?;

            // Take next tournament_size elements for second parent
            let parent2 = candidates[tournament_size..(tournament_size * 2)]
                .iter()
                .max_by(|a, b| a.must_fitness().partial_cmp(&b.must_fitness()).unwrap())
                .ok_or(Error::NoValidParents)?;

            pairs.push((parent1.clone(), parent2.clone())); // No need to clone here anymore
        }

        Ok(pairs)
    }

    #[instrument(level = "debug", skip(self, request), fields(request_id = %request.id, num_offspring = num_offspring, tournament_size = tournament_size, sample_size = sample_size, next_generation_id = next_generation_id))]
    async fn breed_new_individuals(
        &self,
        request: &Request,
        num_offspring: usize,
        tournament_size: usize,
        sample_size: usize,
        next_generation_id: i32, // Same or incremented based on strategy
    ) -> Result<(), Error> {
        // Get candidates for breeding
        let candidates = self
            .genotypes
            .search_genotypes_in_latest_generation(
                sample_size as i64,
                genotypes::Order::Random,
                &genotypes::Filter::default()
                    .with_fitness(true)
                    .with_request_ids(vec![request.id]),
            )
            .await?;

        // Select parents
        let parent_pairs = self.select_by_tournament(num_offspring, tournament_size, candidates)?;

        // Get morphology for mutation bounds
        let morphology = self.morphologies.get_morphology(request.type_hash).await?;

        // Create and mutate new offspring
        let mut new_genotypes = Vec::with_capacity(num_offspring);
        let mut population_updates = Vec::with_capacity(num_offspring);
        let mut events = Vec::with_capacity(num_offspring);

        for (parent1, parent2) in parent_pairs {
            let mut child = self.crossover(&parent1, &parent2, request.id, next_generation_id);

            self.mutate(
                &mut child,
                &morphology,
                request.temperature,
                request.mutation_rate,
            );

            population_updates.push((request.id, child.id));
            events.push(GenotypeGenerated::new(request.id, child.id));
            new_genotypes.push(child);
        }

        // Update database
        let populations = self.populations.clone();
        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    tx_genotypes.new_genotypes(new_genotypes).await?;

                    let mut tx_populations = populations.from_other(tx_genotypes);

                    tx_populations
                        .add_to_population(&population_updates)
                        .await?;

                    let mut publisher = fx_event_bus::Publisher::from_tx(tx_populations);
                    publisher.publish_many(&events).await?;

                    Ok((publisher, ()))
                })
            })
            .await?;

        Ok(())
    }

    #[instrument(level = "debug", skip(self, request), fields(request_id = %request.id, max_evaluations = max_evaluations, population_size = population_size, selection_interval = selection_interval, tournament_size = tournament_size, sample_size = sample_size))]
    async fn maintain_rolling(
        &self,
        request: &Request,
        max_evaluations: u32,
        population_size: u32,
        selection_interval: u32,
        tournament_size: u32,
        sample_size: u32,
    ) -> Result<(), Error> {
        let evaluations = self
            .genotypes
            .get_count_of_genotypes_in_latest_iteration(
                &genotypes::Filter::default()
                    .with_request_ids(vec![request.id])
                    .with_fitness(true),
            )
            .await?;

        if (max_evaluations as i64) > evaluations {
            return Ok(());
        }

        let population_count = self.populations.get_population_count(&request.id).await?;
        if population_count < (population_size - selection_interval) as i64 {
            let current_gen = self.genotypes.get_generation_count(request.id).await?;
            self.breed_new_individuals(
                request,
                selection_interval as usize,
                tournament_size as usize,
                sample_size as usize,
                current_gen,
            )
            .await?;

            return Ok(());
        }

        // Use a compatible repository to open a transaction and publish a termination event
        self.genotypes
            .chain(|tx_genotypes| {
                Box::pin(async move {
                    // Create publisher
                    let mut publisher = fx_event_bus::Publisher::from_tx(tx_genotypes);

                    // Publish termination event
                    publisher
                        .publish(RequestTerminatedEvent::new(request.id))
                        .await?;

                    Ok((publisher, ()))
                })
            })
            .await?;
        Ok(())
    }

    #[instrument(level = "debug", skip(self, request), fields(request_id = %request.id, max_generations = max_generations, population_size = population_size))]
    async fn maintain_generational(
        &self,
        request: Request,
        max_generations: u32,
        population_size: u32,
    ) -> Result<(), Error> {
        let current_gen = self.genotypes.get_generation_count(request.id).await?;

        // Check if we've reached the maximum number of generations
        if max_generations <= current_gen as u32 {
            return Ok(()); // Termination condition: max generations reached
        }

        let population_count = self.populations.get_population_count(&request.id).await?;
        
        // For generational strategy: breed when population is empty (all genotypes evaluated)
        // This happens after all genotypes in the current generation have been evaluated
        if population_count == 0 {
            // Ensure we have evaluated genotypes to breed from
            let evaluated_count = self
                .genotypes
                .get_count_of_genotypes_in_latest_iteration(
                    &genotypes::Filter::default()
                        .with_request_ids(vec![request.id])
                        .with_fitness(true),
                )
                .await?;
            
            // Only breed if we have a full generation of evaluated genotypes
            if evaluated_count >= population_size as i64 {
                self.breed_new_individuals(
                    &request,
                    population_size as usize,
                    (population_size / 8).max(2) as usize, // Larger tournament size for generational
                    population_size as usize,              // Sample from whole population
                    current_gen + 1,                       // Increment generation
                )
                .await?;
            }
        }
        // If population_count > 0, some genotypes are still being evaluated, so wait
        
        Ok(())
    }

    // Shall be run in a job triggered by GenotypeEvaluated
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn maintain_population(&self, request_id: Uuid) -> Result<(), Error> {
        // Get the request
        tracing::debug!("About to fetch request with ID: {}", request_id);
        let request = self.requests.get_request(request_id).await.map_err(|e| {
            tracing::error!("Failed to fetch request with ID {}: {:?}", request_id, e);
            e
        })?;

        // Check for completion
        let top_performer = self
            .genotypes
            .search_genotypes_in_latest_generation(
                1,
                genotypes::Order::Fitness,
                &genotypes::Filter::default()
                    .with_request_ids(vec![request.id])
                    .with_fitness(true),
            )
            .await?;

        if top_performer.is_empty() {
            // Nothing to do
            return Ok(());
        }

        let fitness = top_performer[0].must_fitness();

        if request.is_completed(fitness) {
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
            requests::Strategy::Generational {
                max_generations,
                population_size,
            } => {
                self.maintain_generational(request, max_generations, population_size)
                    .await
            }
            requests::Strategy::Rolling {
                max_evaluations,
                population_size,
                selection_interval,
                tournament_size,
                sample_size,
            } => {
                self.maintain_rolling(
                    &request,
                    max_evaluations,
                    population_size,
                    selection_interval,
                    tournament_size,
                    sample_size,
                )
                .await
            }
        }
    }
}
