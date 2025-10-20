use crate::models::{Genotype, Morphology};
use rand::Rng;
use serde::{Deserialize, Serialize};

fn decay_linear(upper: f64, lower: f64, progress: f64, multiplier: f64) -> f64 {
    lower + (upper - lower) * (1.0 - progress * multiplier).max(0.0)
}

fn decay_exponential(upper: f64, lower: f64, progress: f64, multiplier: f64, exponent: i32) -> f64 {
    lower + (upper - lower) * (1.0 - progress * multiplier).max(0.0).powi(exponent)
}

// ============================================================
// Decay
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decay {
    Constant,
    Linear {
        lower: f64,
        multiplier: f64,
    },
    Exponential {
        lower: f64,
        multiplier: f64,
        exponent: i32,
    },
}

impl Decay {
    fn apply(&self, upper: f64, progress: f64) -> f64 {
        match self {
            Decay::Constant => upper,
            Decay::Linear { lower, multiplier } => {
                decay_linear(upper, *lower, progress, *multiplier)
            }
            Decay::Exponential {
                lower,
                multiplier,
                exponent,
            } => decay_exponential(upper, *lower, progress, *multiplier, *exponent),
        }
    }
}

// ============================================================
// Temperature
// ============================================================
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

        Ok(Self {
            value,
            decay: Decay::Constant,
        })
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

        Ok(Self {
            value: upper,
            decay: Decay::Linear { lower, multiplier },
        })
    }

    pub fn exponential(
        upper: f64,
        lower: f64,
        multiplier: f64,
        exponent: i32,
    ) -> Result<Self, TemperatureError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;

        if lower > upper {
            return Err(TemperatureError::InvalidBounds { lower, upper });
        }

        Ok(Self {
            value: upper,
            decay: Decay::Exponential {
                lower,
                multiplier,
                exponent,
            },
        })
    }

    fn get(&self, progress: f64) -> f64 {
        self.decay.apply(self.value, progress)
    }
}

// ============================================================
// MutationRate
// ============================================================
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

        Ok(Self {
            value,
            decay: Decay::Constant,
        })
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

        Ok(Self {
            value: upper,
            decay: Decay::Linear { lower, multiplier },
        })
    }

    pub fn exponential(
        upper: f64,
        lower: f64,
        multiplier: f64,
        exponent: i32,
    ) -> Result<Self, MutationRateError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;

        if lower > upper {
            return Err(MutationRateError::InvalidBounds { lower, upper });
        }

        Ok(Self {
            value: upper,
            decay: Decay::Exponential {
                lower,
                multiplier,
                exponent,
            },
        })
    }

    fn get(&self, progress: f64) -> f64 {
        self.decay.apply(self.value, progress)
    }
}

// ============================================================
// Mutagen
// ============================================================
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
        Self {
            temperature,
            mutation_rate,
        }
    }

    /// Helper method for backward compatibility - use Mutagen::new() with Temperature/MutationRate constructors instead
    pub fn constant(
        temperature_value: f64,
        mutation_rate_value: f64,
    ) -> Result<Self, MutagenError> {
        Ok(Self::new(
            Temperature::constant(temperature_value)?,
            MutationRate::constant(mutation_rate_value)?,
        ))
    }

    pub(crate) fn mutate<R: Rng>(
        &self,
        rng: &mut R,
        genotype: &mut Genotype,
        morphology: &Morphology,
        progress: f64,
    ) {
        let temperature = self.temperature.get(progress);
        let mutation_rate = self.mutation_rate.get(progress);

        for (gene, bounds) in genotype
            .genome
            .iter_mut()
            .zip(morphology.gene_bounds.iter())
        {
            // Should we mutate this gene?
            if rng.random_range(0.0..1.0) < mutation_rate {
                // Temperature controls mutation step: higher = larger jumps
                let max_step = (1.0 + (bounds.steps() as f64 * temperature)) as i64;

                // Choose direction and step size
                let direction = if rng.random_bool(0.5) { 1 } else { -1 };
                let step = rng.random_range(1..=max_step);

                // Apply mutation and clamp
                *gene = (*gene + direction * step).clamp(0, bounds.steps() as i64 - 1);
            }
        }

        // Recompute hash after mutation since the genome may have changed
        genotype.genome_hash = Genotype::compute_genome_hash(&genotype.genome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GeneBounds;
    use rand::{SeedableRng, rngs::StdRng};
    use uuid::Uuid;

    fn get_test_genotype() -> Genotype {
        Genotype::new("TestType", 123, vec![5, 2], Uuid::now_v7(), 1)
    }

    fn get_test_morphology() -> Morphology {
        Morphology::new(
            "TestType",
            123,
            vec![
                GeneBounds::integer(0, 9, 10).unwrap(), // Gene can be 0-9 (steps=10)
                GeneBounds::integer(0, 4, 5).unwrap(),  // Gene can be 0-4 (steps=5)
            ],
        )
    }

    #[test]
    fn it_validates_temperature_and_mutation_rate_bounds() {
        // Temperature validation
        assert!(Temperature::constant(-0.1).is_err());
        assert!(Temperature::constant(1.5).is_err());
        assert!(Temperature::linear(-0.1, 0.0, 1.0).is_err());
        assert!(Temperature::exponential(1.5, 0.0, 1.0, 2).is_err());

        // MutationRate validation
        assert!(MutationRate::constant(-0.1).is_err());
        assert!(MutationRate::constant(1.5).is_err());
        assert!(MutationRate::linear(-0.1, 0.0, 1.0).is_err());
        assert!(MutationRate::exponential(1.5, 0.0, 1.0, 2).is_err());

        // Lower > upper validation for MutationRate
        assert!(MutationRate::linear(0.3, 0.5, 1.0).is_err());
        assert!(MutationRate::exponential(0.2, 0.4, 1.0, 2).is_err());

        // Lower > upper validation for Temperature
        assert!(Temperature::linear(0.3, 0.5, 1.0).is_err());
        assert!(Temperature::exponential(0.2, 0.4, 1.0, 2).is_err());
    }

    #[test]
    fn it_applies_constant_decay() {
        let temp = Temperature::constant(0.7).unwrap();
        let rate = MutationRate::constant(0.3).unwrap();

        // Constant values shouldn't change with progress
        assert_eq!(temp.get(0.0), 0.7);
        assert_eq!(temp.get(0.5), 0.7);
        assert_eq!(temp.get(1.0), 0.7);

        assert_eq!(rate.get(0.0), 0.3);
        assert_eq!(rate.get(0.5), 0.3);
        assert_eq!(rate.get(1.0), 0.3);
    }

    #[test]
    fn it_applies_linear_decay() {
        let temp = Temperature::linear(1.0, 0.0, 1.0).unwrap();
        let rate = MutationRate::linear(1.0, 0.0, 1.0).unwrap();

        // Linear decay: value * (1.0 - progress * multiplier)
        assert_eq!(temp.get(0.0), 1.0); // No progress
        assert_eq!(temp.get(0.5), 0.5); // Half progress
        assert_eq!(temp.get(1.0), 0.0); // Full progress
        assert_eq!(temp.get(1.5), 0.0); // Over-progress (clamped)

        assert_eq!(rate.get(0.0), 1.0);
        assert_eq!(rate.get(0.5), 0.5);
        assert_eq!(rate.get(1.0), 0.0);
        assert_eq!(rate.get(1.5), 0.0);
    }

    #[test]
    fn it_applies_exponential_decay() {
        let temp = Temperature::exponential(1.0, 0.0, 1.0, 2).unwrap();
        let rate = MutationRate::exponential(1.0, 0.0, 1.0, 2).unwrap();

        // Exponential decay: value * (1.0 - progress * multiplier)^exponent
        assert_eq!(temp.get(0.0), 1.0); // No progress
        assert_eq!(temp.get(0.5), 0.25); // Quadratic: (1.0 - 0.5)^2 = 0.25
        assert_eq!(temp.get(1.0), 0.0); // Full progress

        assert_eq!(rate.get(0.0), 1.0);
        assert_eq!(rate.get(0.5), 0.25);
        assert_eq!(rate.get(1.0), 0.0);
    }

    #[test]
    fn it_mutates() {
        let mut rng = StdRng::seed_from_u64(42);
        let morphology = get_test_morphology();
        let mut genotype = get_test_genotype();

        // Create test data
        let mutagen = Mutagen::new(
            Temperature::constant(0.1).expect("temperature is in range"), // Low temp = small steps
            MutationRate::constant(1.0).expect("mutation_rate is in range"), // 100% mutation rate
        );

        let original_genome = genotype.genome.clone();

        // Mutate with 100% rate - should change something
        mutagen.mutate(&mut rng, &mut genotype, &morphology, 0.0);

        // With 100% mutation rate and seeded RNG, genome should change
        assert_ne!(genotype.genome, original_genome);

        // Verify genes stay within bounds
        assert!(genotype.genome[0] >= 0 && genotype.genome[0] < 10);
        assert!(genotype.genome[1] >= 0 && genotype.genome[1] < 5);
    }

    #[test]
    fn it_respects_zero_mutation_rate() {
        let mut rng = StdRng::seed_from_u64(42);
        let morphology = get_test_morphology();
        let mut genotype = get_test_genotype();

        let mutagen = Mutagen::new(
            Temperature::constant(1.0).expect("temperature is in range"),
            MutationRate::constant(0.0).expect("mutation_rate is in range"), // 0% mutation rate
        );

        let original_genome = genotype.genome.clone();

        mutagen.mutate(&mut rng, &mut genotype, &morphology, 0.0);

        // Should be unchanged with 0% mutation rate
        assert_eq!(genotype.genome, original_genome);
    }

    #[test]
    fn it_composes_different_decay_strategies() {
        // Test that we can mix different decay types
        let mutagen = Mutagen::new(
            Temperature::linear(0.8, 0.1, 0.9).unwrap(), // Linear temperature decay
            MutationRate::exponential(0.5, 0.0, 1.0, 2).unwrap(), // Exponential mutation rate decay
        );

        // Verify the composition works correctly
        assert_eq!(mutagen.temperature.get(0.0), 0.8);
        assert_eq!(mutagen.mutation_rate.get(0.0), 0.5);
        // Exponential decay: lower + (upper - lower) * (1.0 - progress * multiplier)^exponent
        // 0.0 + (0.5 - 0.0) * (1.0 - 0.5 * 1.0)^2 = 0.5 * (0.5)^2 = 0.5 * 0.25 = 0.125
        assert_eq!(mutagen.mutation_rate.get(0.5), 0.125);

        // Linear temperature decay: lower + (upper - lower) * (1.0 - progress * multiplier)
        // 0.1 + (0.8 - 0.1) * (1.0 - 0.5 * 0.9) = 0.1 + 0.7 * 0.55 = 0.1 + 0.385 = 0.485
        assert!((mutagen.temperature.get(0.5) - 0.485).abs() < 1e-10);
    }

    #[test]
    fn it_validates_mutagen_constant_parameters() {
        // Test temperature validation error
        assert!(Mutagen::constant(-0.1, 0.5).is_err());
        assert!(Mutagen::constant(1.5, 0.5).is_err());

        // Test mutation rate validation error
        assert!(Mutagen::constant(0.5, -0.1).is_err());
        assert!(Mutagen::constant(0.5, 1.5).is_err());

        // Test successful case
        assert!(Mutagen::constant(0.5, 0.3).is_ok());
    }
}
