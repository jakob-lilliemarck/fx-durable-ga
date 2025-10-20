use super::GeneBounds;
use const_fnv1a_hash::fnv1a_hash_str_32;

/// Trait for types that can be encoded as genomes for genetic algorithm optimization.
/// Provides the interface between domain-specific types and the GA engine.
pub trait Encodeable {
    /// Unique name identifier for this type.
    const NAME: &str;
    /// Hash derived from the name for efficient type identification.
    const HASH: i32 = fnv1a_hash_str_32(Self::NAME) as i32;

    /// The decoded phenotype that will be passed to evaluators.
    type Phenotype;

    /// Returns the gene bounds that define the search space structure.
    fn morphology() -> Vec<GeneBounds>;
    /// Encodes an instance of this type into a genome representation.
    fn encode(&self) -> Vec<i64>;
    /// Decodes a genome into the phenotype for evaluation.
    fn decode(genes: &[i64]) -> Self::Phenotype;
}
