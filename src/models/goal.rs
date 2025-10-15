use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FitnessGoal {
    Minimize { threshold: f64 },
    Maximize { threshold: f64 },
}

#[derive(Debug, thiserror::Error)]
#[error("fitness goal threshold must be between 0.0 and 1.0, got {0}")]
pub struct ThresholdOutOfRange(f64);

impl FitnessGoal {
    pub fn minimize(threshold: f64) -> Result<Self, ThresholdOutOfRange> {
        let threshold = Self::validate(threshold)?;

        Ok(Self::Minimize { threshold })
    }

    pub fn maximize(threshold: f64) -> Result<Self, ThresholdOutOfRange> {
        let threshold = Self::validate(threshold)?;

        Ok(Self::Maximize { threshold })
    }

    pub(crate) fn is_reached(&self, fitness: f64) -> bool {
        match self {
            FitnessGoal::Minimize { threshold } => fitness <= *threshold,
            FitnessGoal::Maximize { threshold } => fitness >= *threshold,
        }
    }

    fn validate(threshold: f64) -> Result<f64, ThresholdOutOfRange> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(ThresholdOutOfRange(threshold));
        }

        Ok(threshold)
    }
}
