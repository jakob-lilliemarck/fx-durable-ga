use crate::models::{Generation, Phenotype};
use std::error::Error;

pub trait GenerateGeneration<P: Phenotype> {
    type Error: Error;

    fn generate() -> Result<Generation, Self::Error>;
}

pub trait EvaluatePhenotype<P: Phenotype> {
    type Error: Error;

    fn evaluate(phenotype: P) -> Result<f64, Self::Error>;
}

pub trait Optimizable<P: Phenotype>: EvaluatePhenotype<P> + GenerateGeneration<P> {}
