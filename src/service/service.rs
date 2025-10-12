use crate::repositories::chainable::{Chain, FromTxType};

use crate::repositories::{
    genotypes,
    morphologies::{self, Morphology},
    requests::{self, Request},
};
use crate::service::events::{GenotypeEvaluated, GenotypeGenerated, OptimizationRequested};
use const_fnv1a_hash::fnv1a_hash_str_32;
use futures::future::BoxFuture;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("RequestsRepositoryError: {0}")]
    RequestsRepositoryError(#[from] requests::Error),
    #[error("MorphologiesRepositoryError: {0}")]
    MorphologiesRepositoryError(#[from] morphologies::Error),
    #[error("GenotypesRepositoryError: {0}")]
    GenotypesRepositoryError(#[from] genotypes::Error),
    #[error("UnknownType: type_name={type_name}, type_hash={type_hash}")]
    UnknownTypeError { type_hash: i32, type_name: String },
    #[error("EvaluationError: {0}")]
    EvaluationError(#[from] anyhow::Error),
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

trait TypeErasedEvaluator {
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}

struct ErasedEvaluator<P, E: Evaluator<P>> {
    evaluator: E,
    decode: fn(&[i64]) -> P,
}

impl<P, E: Evaluator<P>> TypeErasedEvaluator for ErasedEvaluator<P, E> {
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
        let phenotype = (self.decode)(genes);
        self.evaluator.fitness(phenotype)
    }
}

// optimization service
pub struct Service<'a> {
    requests: requests::Repository,
    morphologies: morphologies::Repository,
    genotypes: genotypes::Repository,
    evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'a>>,
}

pub struct ServiceBuilder<'a> {
    requests: requests::Repository,
    morphologies: morphologies::Repository,
    genotypes: genotypes::Repository,
    evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'a>>,
}

impl<'a> ServiceBuilder<'a> {
    // NOTE:
    // This is a N+1, probably not so bad, it shouldn't be like this.
    // Its simple though, so I'll go with it for a first version
    pub async fn register<T, E>(mut self, evaluator: E) -> Result<Self, Error>
    where
        T: Encodeable + 'a,
        E: Evaluator<T::Phenotype> + 'a,
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

    pub fn build(self) -> Service<'a> {
        Service {
            requests: self.requests,
            morphologies: self.morphologies,
            genotypes: self.genotypes,
            evaluators: self.evaluators,
        }
    }
}

impl<'a> Service<'a> {
    pub fn builder(
        requests: requests::Repository,
        morphologies: morphologies::Repository,
        genotypes: genotypes::Repository,
    ) -> ServiceBuilder<'a> {
        ServiceBuilder {
            requests,
            morphologies,
            genotypes,
            evaluators: HashMap::new(),
        }
    }

    pub async fn new_optimization_request(
        &self,
        type_name: &str,
        type_hash: i32,
        goal: requests::FitnessGoal,
        threshold: f64,
        strategy: requests::Strategy,
    ) -> Result<(), Error> {
        self.requests
            .chain(|mut tx_requests| {
                Box::pin(async move {
                    // Create a new optimization request with the repository
                    let request = tx_requests
                        .new_request(Request::new(
                            type_name, type_hash, goal, threshold, strategy,
                        ))
                        .await?;

                    // Instantiate a publisher
                    let mut publisher = fx_event_bus::Publisher::from_other(tx_requests);

                    // Publish an event within the same transaction
                    publisher
                        .publish(OptimizationRequested::new(request.id))
                        .await?;

                    Ok((publisher, request))
                })
            })
            .await?;

        Ok(())
    }

    // Shall be run in a job triggered by OptimizationRequested
    pub async fn generate_initial_population(
        &self,
        request_id: Uuid,
        generation_id: i32,
    ) -> Result<(), Error> {
        // Get the optimization request
        let request = self.requests.get_request(request_id).await?;

        // Get the morphology of the type under optimization
        let morphology = self.morphologies.get_morphology(request.type_hash).await?;

        let mut rng = rand::rng();
        let mut genotypes = Vec::with_capacity(request.population_size() as usize);
        let mut population = Vec::with_capacity(request.population_size() as usize);
        let mut events = Vec::with_capacity(request.population_size() as usize);
        for _ in 0..request.population_size() {
            let genotype = genotypes::Genotype::new(
                &request.type_name,
                request.type_hash,
                morphology.random(&mut rng),
                request.id,
                generation_id,
            );
            population.push((request.id, genotype.id));
            let event = GenotypeGenerated::new(request.id, genotype.id);

            genotypes.push(genotype);
            events.push(event);
        }

        self.genotypes
            .chain(|mut tx_genotypes| {
                Box::pin(async move {
                    // Create the genotypes with the repository
                    tx_genotypes.new_genotypes(genotypes).await?;

                    // Add all to the active population of this request
                    tx_genotypes.add_to_population(&population).await?;

                    // Instantiate a publisher
                    let mut publisher = fx_event_bus::Publisher::from_other(tx_genotypes);

                    // Publish one event for each generated phenotype
                    publisher.publish_many(&events).await?;

                    Ok((publisher, request))
                })
            })
            .await?;

        Ok(())
    }

    // Shall be run in a job triggerd by GenotypeGenerated
    pub async fn evaluate_genotype(
        &self,
        request_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        // Get the genotype from the database
        let genotype = self.genotypes.get_genotype(genotype_id).await?;

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
                    // Update the fitness of the genotype
                    tx_genotypes.set_fitness(genotype_id, fitness).await?;

                    // Remove the genotype from the activde population
                    tx_genotypes
                        .remove_from_population(request_id, genotype_id)
                        .await?;

                    // Instantiate a publisher
                    let mut publisher = fx_event_bus::Publisher::from_other(tx_genotypes);

                    // Publish an event
                    publisher
                        .publish(GenotypeEvaluated::new(request_id, genotype_id))
                        .await?;

                    Ok((publisher, ()))
                })
            })
            .await?;

        Ok(())
    }

    async fn maintain_generational(
        &self,
        request: Request,
        max_generations: u32,
        population_size: u32,
    ) -> Result<(), Error> {
        let genotypes = self
            .genotypes
            .search_genotypes_in_latest_generation(
                population_size as i64,
                &genotypes::Filter::default().with_fitness(true),
            )
            .await?;

        if genotypes.len() < population_size as usize {
            // population_size is not yet reached, return
            return Ok(());
        }

        let generation_id = genotypes
            .get(0)
            .map(|g| g.generation_id)
            .expect("genotypes len is equal or larger to population size");

        if max_generations < generation_id as u32 {
            // FIXME!
            //
            // Perform selection and generate the next generation!
            todo!()
        }

        // FIXME!
        //
        // Publish RequestTerminated::new(request.id)
        todo!()
    }

    async fn maintain_rolling(
        &self,
        request: Request,
        max_evaluations: u32,
        population_size: u32,
        selection_interval: u32,
    ) -> Result<(), Error> {
        // Get the count of completed evaluations in the latest generation
        let evaluations = self
            .genotypes
            .count_genotypes_in_latest_iteration(
                &genotypes::Filter::default()
                    .with_request_ids(vec![request.id])
                    .with_fitness(true),
            )
            .await?;

        if (max_evaluations as i64) < evaluations {
            // max_evaluations not yet reached, return
            return Ok(());
        }

        let population_count = self.genotypes.get_population_count(request.id).await?;
        if population_count < (population_size - selection_interval) as i64 {
            // FIXME!
            //
            // Perform selection and generate selection_interval number of new Genotypes!
            todo!()
        }

        // FIXME!
        //
        // Publish RequestTerminated::new(request.id)
        todo!()
    }

    // Shall be run in a job triggered by GenotypeEvaluated
    pub async fn maintain_population(&self, request_id: Uuid) -> Result<(), Error> {
        // Get the request
        let request = self.requests.get_request(request_id).await?;

        // Check for completion
        let top_performer = self
            .genotypes
            .search_genotypes_in_latest_generation(
                1,
                &genotypes::Filter::default()
                    .with_request_ids(vec![request.id])
                    .with_fitness(true),
            )
            .await?;

        if top_performer.is_empty() {
            // Nothing to do
            return Ok(());
        }

        let fitness = top_performer[0]
            .fitness()
            .expect("Filtered out genotypes without fitness in query");

        if request.is_completed(fitness) {
            // FIXME!
            //
            // Publish RequestCompleted::new(request_id)
            return Ok(());
        }

        match request.strategy {
            requests::Strategy::Generational {
                max_generations,
                population_size,
            } => {
                return self
                    .maintain_generational(request, max_generations, population_size)
                    .await;
            }
            requests::Strategy::Rolling {
                max_evaluations,
                population_size,
                selection_interval,
            } => {
                return self
                    .maintain_rolling(
                        request,
                        max_evaluations,
                        population_size,
                        selection_interval,
                    )
                    .await;
            }
        }
    }
}
