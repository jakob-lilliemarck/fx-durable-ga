use serde::{Deserialize, Serialize};

/// Strategy for generating initial populations in genetic algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    LatinHypercube { population_size: u32 },
    Random { population_size: u32 },
}

impl Distribution {
    pub fn latin_hypercube(population_size: u32) -> Self {
        Distribution::LatinHypercube { population_size }
    }

    pub fn random(population_size: u32) -> Self {
        Distribution::Random { population_size }
    }
}
