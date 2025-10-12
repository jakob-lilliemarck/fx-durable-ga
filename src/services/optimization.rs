use crate::{
    gene::Population,
    repositories::{genotypes, populations, requests},
};
use sqlx::PgPool;
use std::any::TypeId;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {}

// optimization service
pub struct Service<'a> {
    requests: requests::Repository,
    genotypes: genotypes::Repository,
    population: populations::Repository,
    registry: crate::registry::Registry<'a>,
}

impl<'a> Service<'a> {
    pub fn new_optimization() -> Result<(), Error> {
        // 1. Write the optimization to the database
        // 2. Publish OptimizationRequested
        todo!()
    }

    pub fn next_population(
        &self,
        optimization_id: Uuid,
        phenotype_id: TypeId,
    ) -> Result<Population, Error> {
        // 1. Get the Morphology under optimization <- this is the user defined morphology
        // 2. Check if there is a prior generation for this optimization
        //  -> if there is, use it to breed a new one
        //  -> otherwise randomize the first population
        // 3. Publish GenotypeGenerated events (for each)
        todo!()
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
