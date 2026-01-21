use rand::Rng;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::Debug;

/// The core trait for any type that can be evolved by the generic framework.
///
/// Types implementing this trait define their own structure, mutation rules,
/// and crossover logic. The framework handles persistence (via Serde) and
/// the evolutionary loop (Selection -> Breeding -> Evaluation).
pub trait Evolvable: Serialize + DeserializeOwned + Clone + Send + Sync + Debug {
    /// Mutate the genome in place.
    ///
    /// # Arguments
    /// * `rng` - The random number generator to use for stochastic operations.
    /// * `mutation_rate` - The global probability of mutation occurring (0.0 - 1.0).
    ///   Types with multiple fields should use this to determine *which* fields to mutate.
    /// * `temperature` - A simulated annealing parameter (0.0 - 1.0) often used to scale
    ///   the *magnitude* of mutation (e.g., small tweaks at low temp, wild changes at high temp).
    fn mutate<R: Rng>(&mut self, rng: &mut R, mutation_rate: f64, temperature: f64);

    /// Create a new offspring by crossing over this parent with another.
    ///
    /// # Arguments
    /// * `other` - The second parent to crossover with.
    /// * `rng` - Random number generator for selecting crossover points/strategies.
    ///
    /// # Returns
    /// A new instance representing the child genome.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self;
}
