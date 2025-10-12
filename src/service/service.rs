use crate::repositories::requests::{Chain, FromTxType, ToTx};
use crate::repositories::{
    genotypes,
    morphologies::{self, Morphology},
    requests::{self, Request},
};
use crate::service::events::OptimizationRequested;
use const_fnv1a_hash::fnv1a_hash_str_32;
use futures::future::BoxFuture;
use fx_mq_jobs::Publisher;
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
    fn fitness<'a>(&self, phenotype: P) -> BoxFuture<'a, Result<f64, String>>;
}

trait TypeErasedEvaluator {
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, String>>;
}

struct ErasedEvaluator<P, E: Evaluator<P>> {
    evaluator: E,
    decode: fn(&[i64]) -> P,
}

impl<P, E: Evaluator<P>> TypeErasedEvaluator for ErasedEvaluator<P, E> {
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, String>> {
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
    // TODO:
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
    ) -> Result<requests::Request, Error> {
        let request = self
            .requests
            .chain(|mut tx_requests| {
                Box::pin(async move {
                    // Create a new optimization request
                    let request = tx_requests
                        .new_request(Request::new(
                            type_name, type_hash, goal, threshold, strategy,
                        ))
                        .await?;

                    let mut publisher = fx_event_bus::Publisher::from_other(tx_requests);

                    // Publish an event within the same transaction
                    publisher
                        .publish(OptimizationRequested::new(request.id))
                        .await?;

                    Ok((publisher, request))
                })
            })
            .await?;

        Ok(request)
    }

    pub fn evaluate_phenotype(
        &self,
        optimization_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        // 1. Get the Genotype
        // 2. Pass it to Registry::evaluate <- this is the long running user defined function
        // 3. Write a fitness score to the repository
        // 4. Publish PhenotypeEvaluated event
        todo!()
    }

    pub fn next_population(&self, optimization_id: Uuid, type_hash: i32) -> Result<(), Error> {
        // 1. Get the Morphology under optimization <- this is the user defined morphology
        // 2. Check if there is a prior generation for this optimization
        //  -> if there is, use it to breed a new one
        //  -> otherwise randomize the first population
        // 3. Publish GenotypeGenerated events (for each)
        todo!()
    }

    pub fn evaluate_population(
        &self,
        optimization_id: Uuid,
        population_id: Uuid,
    ) -> Result<(), Error> {
        // 1. Get genotypes of this population
        // 2. If any miss a fitness value, return
        //
        // 3. Get the Optimization
        // 4. Get the Population
        // 5. Get the top performing Genotype
        // 6. If the generation number is equal to the max number of generations or if the top performer meets the goal
        //  -> Publish OptimizationCompleted event
        //  -> Otherwise publish GenerationCompleted event
        todo!()
    }
}
