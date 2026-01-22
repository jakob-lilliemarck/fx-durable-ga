use futures::future::BoxFuture;
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::fmt::Debug;
use anyhow::Result;

/// The core trait for any type that can be evolved by the generic framework.
///
/// Types implementing this trait define their own structure, mutation rules,
/// and crossover logic. The framework handles persistence (via Serde) and
/// the evolutionary loop (Selection -> Breeding -> Evaluation).

/// A type-erased manager for a specific `Evolvable` genotype.
///
/// This is the trait the framework interacts with via a registry. Implementors of this
/// trait provide the bridge between the framework's generic `serde_json::Value` representation
/// and the user's concrete `Evolvable` type.
pub trait GenotypeManager: Send + Sync {
    /// Generate a new, random genome as a JSON Value.
    fn random(&self, rng: &mut impl Rng) -> Value;

    /// Perform crossover on two JSON values, returning a new JSON child.
    fn crossover(&self, parent1: &Value, parent2: &Value, rng: &mut impl Rng) -> Result<Value>;

    /// Mutate a JSON genome in place.
    fn mutate(&self, genotype: &mut Value, rng: &mut impl Rng, mutation_rate: f64, temperature: f64) -> Result<()>;

    /// Evaluate the fitness of a JSON genome.
    fn evaluate<'a>(&'a self, genotype: &'a Value) -> BoxFuture<'a, Result<f64>>;
}
