use crate::models::{Genotype, Morphology};
use rand::Rng;
use serde::{Deserialize, Serialize};

fn decay_linear(value: f64, progress: f64, multiplier: f64) -> f64 {
    value * (1.0 - progress * multiplier).max(0.0)
}

fn decay_exponential(value: f64, progress: f64, multiplier: f64, exponent: i32) -> f64 {
    value * (1.0 - progress * multiplier).max(0.0).powi(exponent)
}

// ============================================================
// Decay
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decay {
    Constant,
    Linear { multiplier: f64 },
    Exponential { multiplier: f64, exponent: i32 },
}

impl Decay {
    fn apply(&self, value: f64, progress: f64) -> f64 {
        match self {
            Decay::Constant => value,
            Decay::Linear { multiplier } => decay_linear(value, progress, *multiplier),
            Decay::Exponential {
                multiplier,
                exponent,
            } => decay_exponential(value, progress, *multiplier, *exponent),
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
#[error("temperature must be between 0.0 and 1.0, got: {0}")]
pub struct TemperatureOutOfRange(f64);

impl Temperature {
    pub fn new(value: f64, decay: Decay) -> Result<Self, TemperatureOutOfRange> {
        let value = Self::validate(value)?;

        Ok(Self { value, decay })
    }

    pub fn constant(value: f64) -> Result<Self, TemperatureOutOfRange> {
        let value = Self::validate(value)?;

        Ok(Self {
            value,
            decay: Decay::Constant,
        })
    }

    fn validate(value: f64) -> Result<f64, TemperatureOutOfRange> {
        if !(0.0..=1.0).contains(&value) {
            return Err(TemperatureOutOfRange(value));
        }

        Ok(value)
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
#[error("mutation_rate must be between 0.0 and 1.0, got: {0}")]
pub struct MutationRateOutOfRange(f64);

impl MutationRate {
    pub fn new(value: f64, decay: Decay) -> Result<Self, MutationRateOutOfRange> {
        let value = Self::validate(value)?;

        Ok(Self { value, decay })
    }

    pub fn constant(value: f64) -> Result<Self, MutationRateOutOfRange> {
        let value = Self::validate(value)?;

        Ok(Self {
            value,
            decay: Decay::Constant,
        })
    }

    fn validate(value: f64) -> Result<f64, MutationRateOutOfRange> {
        if !(0.0..=1.0).contains(&value) {
            return Err(MutationRateOutOfRange(value));
        }

        Ok(value)
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
    MutationRate(#[from] MutationRateOutOfRange),
    #[error("Temperature error: {0}")]
    Temperature(#[from] TemperatureOutOfRange),
}

impl Mutagen {
    pub fn new(temperature: Temperature, mutation_rate: MutationRate) -> Self {
        Self {
            temperature,
            mutation_rate,
        }
    }

    pub fn constant(
        temperature_value: f64,
        mutation_rate_value: f64,
    ) -> Result<Self, MutagenError> {
        let temperature = Temperature::constant(temperature_value)?;
        let mutation_rate = MutationRate::constant(mutation_rate_value)?;

        Ok(Self {
            temperature,
            mutation_rate,
        })
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
    fn test_temperature_validation_errors() {
        assert!(Temperature::new(-0.1, Decay::Constant).is_err());
        assert!(Temperature::new(1.5, Decay::Constant).is_err());
        assert!(Temperature::constant(-0.1).is_err());
        assert!(Temperature::constant(1.5).is_err());
    }

    #[test]
    fn test_mutation_rate_validation_errors() {
        assert!(MutationRate::new(-0.1, Decay::Constant).is_err());
        assert!(MutationRate::new(1.5, Decay::Constant).is_err());
        assert!(MutationRate::constant(-0.1).is_err());
        assert!(MutationRate::constant(1.5).is_err());
    }

    #[test]
    fn test_mutagen_validation_errors() {
        let result = Mutagen::constant(-0.1, 0.5); // Invalid temperature
        assert!(result.is_err());

        let result = Mutagen::constant(0.5, -0.1); // Invalid mutation rate
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_decay_through_temperature() {
        let temp = Temperature::new(1.0, Decay::Linear { multiplier: 1.0 }).unwrap();

        assert_eq!(temp.get(0.0), 1.0); // No progress
        assert_eq!(temp.get(0.5), 0.5); // Half progress
        assert_eq!(temp.get(1.0), 0.0); // Full progress
        assert_eq!(temp.get(1.5), 0.0); // Over-progress (clamped)
    }

    #[test]
    fn test_exponential_decay_through_temperature() {
        let temp = Temperature::new(
            1.0,
            Decay::Exponential {
                multiplier: 1.0,
                exponent: 2,
            },
        )
        .unwrap();

        assert_eq!(temp.get(0.0), 1.0); // No progress
        assert_eq!(temp.get(0.5), 0.25); // Quadratic: (1.0 - 0.5)^2 = 0.25
        assert_eq!(temp.get(1.0), 0.0); // Full progress
    }

    #[test]
    fn test_linear_decay_through_mutation_rate() {
        let rate = MutationRate::new(1.0, Decay::Linear { multiplier: 1.0 }).unwrap();

        assert_eq!(rate.get(0.0), 1.0); // No progress
        assert_eq!(rate.get(0.5), 0.5); // Half progress
        assert_eq!(rate.get(1.0), 0.0); // Full progress
        assert_eq!(rate.get(1.5), 0.0); // Over-progress (clamped)
    }

    #[test]
    fn test_exponential_decay_through_mutation_rate() {
        let rate = MutationRate::new(
            1.0,
            Decay::Exponential {
                multiplier: 1.0,
                exponent: 2,
            },
        )
        .unwrap();

        assert_eq!(rate.get(0.0), 1.0); // No progress
        assert_eq!(rate.get(0.5), 0.25); // Quadratic: (1.0 - 0.5)^2 = 0.25
        assert_eq!(rate.get(1.0), 0.0); // Full progress
    }

    #[test]
    fn test_constant_decay() {
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
    fn it_mutates() {
        let mut rng = StdRng::seed_from_u64(42);
        let morphology = get_test_morphology();
        let mut genotype = get_test_genotype();

        // Create test data
        let mutagen = Mutagen::new(
            Temperature::new(0.1, Decay::Constant).expect("temperature is in range"), // Low temp = small steps
            MutationRate::new(1.0, Decay::Constant).expect("mutation_rate is in range"), // 100% mutation rate
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
            Temperature::new(1.0, Decay::Constant).expect("temperature is in range"),
            MutationRate::new(0.0, Decay::Constant).expect("mutation_rate is in range"), // 0% mutation rate
        );

        let original_genome = genotype.genome.clone();

        mutagen.mutate(&mut rng, &mut genotype, &morphology, 0.0);

        // Should be unchanged with 0% mutation rate
        assert_eq!(genotype.genome, original_genome);
    }

    #[test]
    fn it_applies_progress_to_decay() {
        let mutagen = Mutagen::new(
            Temperature::new(1.0, Decay::Linear { multiplier: 1.0 })
                .expect("temperature is in range"),
            MutationRate::new(0.8, Decay::Constant).expect("mutation_range is in range"),
        );

        // Test that temperature decays with progress
        assert_eq!(mutagen.temperature.get(0.0), 1.0);
        assert_eq!(mutagen.temperature.get(0.5), 0.5);
        assert_eq!(mutagen.temperature.get(1.0), 0.0);

        // Test that constant rate doesn't decay
        assert_eq!(mutagen.mutation_rate.get(0.5), 0.8);
    }
}
