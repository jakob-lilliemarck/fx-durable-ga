use crate::models::{Genotype, TypeName};

pub struct EncodeInput {
    pub values: Vec<f32>,
    pub dimensions: Vec<usize>,
}

/// Something that can be indexed by the indexing service
pub trait GenotypeIndexer: TypeName + Send + Sync {
    fn input(&self, genotype: &Genotype) -> EncodeInput;

    /// hash() is intended to capture a hash value of the indexing context.
    /// For example, a hash of the dataset or configuration options used to produce `EncodeInput`.
    /// The hash can be used to determine if two embeddings can be meaningfully compared or not.
    fn context_hash(&self) -> String;
}
