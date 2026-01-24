use serde::{Deserialize, Serialize};
use tracing::instrument;

fn decay_linear(upper: f64, lower: f64, progress: f64, multiplier: f64) -> f64 {
    lower + (upper - lower) * (1.0 - progress * multiplier).max(0.0)
}

fn decay_exponential(upper: f64, lower: f64, progress: f64, multiplier: f64, exponent: i32) -> f64 {
    lower + (upper - lower) * (1.0 - progress * multiplier).max(0.0).powi(exponent)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decay {
    Constant,
    Linear { lower: f64, multiplier: f64 },
    Exponential { lower: f64, multiplier: f64, exponent: i32 },
}

impl Decay {
    fn apply(&self, upper: f64, progress: f64) -> f64 {
        match self {
            Decay::Constant => upper,
            Decay::Linear { lower, multiplier } => decay_linear(upper, *lower, progress, *multiplier),
            Decay::Exponential { lower, multiplier, exponent } => {
                decay_exponential(upper, *lower, progress, *multiplier, *exponent)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Temperature {
    value: f64,
    decay: Decay,
}

#[derive(Debug, thiserror::Error)]
pub enum TemperatureError {
    #[error("temperature value must be between 0.0 and 1.0, got: {0}")]
    ValueOutOfRange(f64),
    #[error("temperature lower bound ({lower}) must be <= upper bound ({upper})")]
    InvalidBounds { lower: f64, upper: f64 },
}

impl Temperature {
    pub fn constant(value: f64) -> Result<Self, TemperatureError> {
        let value = Self::validate(value)?;
        Ok(Self { value, decay: Decay::Constant })
    }

    fn validate(value: f64) -> Result<f64, TemperatureError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(TemperatureError::ValueOutOfRange(value));
        }
        Ok(value)
    }

    pub fn linear(upper: f64, lower: f64, multiplier: f64) -> Result<Self, TemperatureError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;
        if lower > upper {
            return Err(TemperatureError::InvalidBounds { lower, upper });
        }
        Ok(Self { value: upper, decay: Decay::Linear { lower, multiplier } })
    }

    pub fn exponential(upper: f64, lower: f64, multiplier: f64, exponent: i32) -> Result<Self, TemperatureError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;
        if lower > upper {
            return Err(TemperatureError::InvalidBounds { lower, upper });
        }
        Ok(Self { value: upper, decay: Decay::Exponential { lower, multiplier, exponent } })
    }

    pub(crate) fn at_progress(&self, progress: f64) -> f64 {
        self.decay.apply(self.value, progress)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRate {
    value: f64,
    decay: Decay,
}

#[derive(Debug, thiserror::Error)]
pub enum MutationRateError {
    #[error("mutation rate value must be between 0.0 and 1.0, got: {0}")]
    ValueOutOfRange(f64),
    #[error("mutation rate lower bound ({lower}) must be <= upper bound ({upper})")]
    InvalidBounds { lower: f64, upper: f64 },
}

impl MutationRate {
    pub fn constant(value: f64) -> Result<Self, MutationRateError> {
        let value = Self::validate(value)?;
        Ok(Self { value, decay: Decay::Constant })
    }

    fn validate(value: f64) -> Result<f64, MutationRateError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(MutationRateError::ValueOutOfRange(value));
        }
        Ok(value)
    }

    pub fn linear(upper: f64, lower: f64, multiplier: f64) -> Result<Self, MutationRateError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;
        if lower > upper {
            return Err(MutationRateError::InvalidBounds { lower, upper });
        }
        Ok(Self { value: upper, decay: Decay::Linear { lower, multiplier } })
    }

    pub fn exponential(upper: f64, lower: f64, multiplier: f64, exponent: i32) -> Result<Self, MutationRateError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;
        if lower > upper {
            return Err(MutationRateError::InvalidBounds { lower, upper });
        }
        Ok(Self { value: upper, decay: Decay::Exponential { lower, multiplier, exponent } })
    }

    pub(crate) fn at_progress(&self, progress: f64) -> f64 {
        self.decay.apply(self.value, progress)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutagen {
    mutation_rate: MutationRate,
    temperature: Temperature,
}

#[derive(Debug, thiserror::Error)]
pub enum MutagenError {
    #[error("Mutation rate error: {0}")]
    MutationRate(#[from] MutationRateError),
    #[error("Temperature error: {0}")]
    Temperature(#[from] TemperatureError),
}

impl Mutagen {
    pub fn new(temperature: Temperature, mutation_rate: MutationRate) -> Self {
        Self { temperature, mutation_rate }
    }

    pub fn constant(temperature_value: f64, mutation_rate_value: f64) -> Result<Self, MutagenError> {
        Ok(Self::new(
            Temperature::constant(temperature_value)?,
            MutationRate::constant(mutation_rate_value)?,
        ))
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn temperature(&self) -> &Temperature {
        &self.temperature
    }
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn mutation_rate(&self) -> &MutationRate {
        &self.mutation_rate
    }
}
