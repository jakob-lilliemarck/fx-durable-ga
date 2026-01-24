use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crossover {
    probability: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum CrossoverError {
    #[error("crossover probability must be between 0.0 and 1.0, got {0}")]
    ProbabilityOutOfRange(f64),
}

impl Crossover {
    pub fn uniform(probability: f64) -> Result<Self, CrossoverError> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(CrossoverError::ProbabilityOutOfRange(probability));
        }
        Ok(Self { probability })
    }

    pub fn single_point() -> Result<Self, CrossoverError> {
        Ok(Self { probability: 0.5 })
    }

    pub fn probability(&self) -> f64 {
        self.probability
    }
}
