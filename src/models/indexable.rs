use crate::services::indexing::EncodeInput;
use const_fnv1a_hash::fnv1a_hash_str_32;

/// Something that can be indexed by the indexing service
pub trait Indexable: Send + Sync {
    fn name(&self) -> &str;

    fn hash(&self) -> i32 {
        fnv1a_hash_str_32(self.name()) as i32
    }

    fn encode_inputs(&self) -> Vec<EncodeInput>;
}
