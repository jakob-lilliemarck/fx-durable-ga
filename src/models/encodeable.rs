use super::GeneBounds;
use const_fnv1a_hash::fnv1a_hash_str_32;

pub trait Encodeable {
    const NAME: &str;
    const HASH: i32 = fnv1a_hash_str_32(Self::NAME) as i32;

    type Phenotype;

    fn morphology() -> Vec<GeneBounds>;
    fn encode(&self) -> Vec<i64>;
    fn decode(genes: &[i64]) -> Self::Phenotype;
}
