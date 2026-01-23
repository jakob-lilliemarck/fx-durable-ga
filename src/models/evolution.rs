use anyhow::Result;
use const_fnv1a_hash::fnv1a_hash_str_32;
use futures::future::BoxFuture;
use rand::Rng;
use serde_json::Value;

/// A type-erased manager for a specific genotype.
///
/// This is the trait the framework interacts with via a registry. Implementors of this
/// trait provide the bridge between the framework's generic `serde_json::Value` representation
/// and the user's concrete genotype type.
pub trait GenotypeManager: Send + Sync {
    /// Unique name identifier for the type being managed.
    fn name(&self) -> &'static str;

    /// Hash derived from the name for efficient type identification.
    /// Default implementation provided using FNV-1a.
    fn hash(&self) -> i32 {
        fnv1a_hash_str_32(self.name()) as i32
    }

    /// Generate a new, random genome as a JSON Value.
    fn random(&self, rng: &mut impl Rng) -> Value;

    /// Perform crossover on two JSON values, returning a new JSON child.
    fn crossover(&self, parent1: &Value, parent2: &Value, rng: &mut impl Rng) -> Result<Value>;

    /// Mutate a JSON genome in place.
    fn mutate(&self, genotype: &mut Value, rng: &mut impl Rng, mutation_rate: f64, temperature: f64) -> Result<()>;

    /// Evaluate the fitness of a JSON genome.
    fn evaluate<'a>(&'a self, genotype: &'a Value) -> BoxFuture<'a, Result<f64>>;
}
