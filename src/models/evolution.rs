use anyhow::Result;
use const_fnv1a_hash::fnv1a_hash_str_32;
use futures::future::BoxFuture;
use rand::RngCore;
use serde_json::Value;

pub trait TypeName {
    /// Unique name identifier for the type
    fn type_name(&self) -> &'static str;

    /// Hash derived from the name for efficient type identification.
    /// Default implementation provided using FNV-1a.
    fn type_hash(&self) -> i32 {
        fnv1a_hash_str_32(self.type_name()) as i32
    }
}

/// A type-erased manager for a specific genotype.
///
/// This is the trait the framework interacts with via a registry. Implementors of this
/// trait provide the bridge between the framework's generic `serde_json::Value` representation
/// and the user's concrete genotype type.
pub trait GenotypeManager: TypeName + Send + Sync {
    /// Generate a new, random genome as a JSON Value.
    fn random(&self, rng: &mut dyn RngCore, user_defined: &Value) -> Result<Value>;

    /// Perform crossover on two JSON values, returning a new JSON child.
    fn crossover(
        &self,
        parent1: &Value,
        parent2: &Value,
        rng: &mut dyn RngCore,
        user_defined: &Value,
    ) -> Result<Value>;

    /// Mutate a JSON genome in place.
    fn mutate(
        &self,
        genotype: &mut Value,
        rng: &mut dyn RngCore,
        user_defined: &Value,
    ) -> Result<()>;

    /// Evaluate the fitness of a JSON genome.
    fn evaluate<'a>(
        &'a self,
        genotype: &'a Value,
        user_defined: &'a Value,
    ) -> BoxFuture<'a, Result<f64>>;
}
