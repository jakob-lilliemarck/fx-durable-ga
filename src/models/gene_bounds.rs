use super::Gene;
use rand::{Rng, rngs::ThreadRng};
use serde::{Deserialize, Serialize};
use tracing::instrument;

#[derive(Debug, thiserror::Error)]
pub enum GeneBoundError {
    #[error(
        "InvalidBounds: lower bound must be smaller than upper. lower = {lower}, upper={upper}"
    )]
    InvalidBound { lower: f64, upper: f64 },
    #[error("StepsOverflow: steps is too large. steps={steps}, max={max}")]
    StepsOverflow { steps: u32, max: i32 },
    #[error("ZeroSteps: number of steps must be greater than 0")]
    ZeroSteps,
}

impl GeneBoundError {
    pub(crate) fn steps_overflow(steps: u32) -> Self {
        Self::StepsOverflow {
            steps,
            max: i32::MAX,
        }
    }

    pub(crate) fn invalid_bound(lower: f64, upper: f64) -> Self {
        Self::InvalidBound { lower, upper }
    }

    pub(crate) fn zero_steps() -> Self {
        Self::ZeroSteps
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct GeneBounds {
    /// Lower bound scaled by scale_factor (for precision)
    pub(crate) lower_scaled: i64,
    /// Upper bound scaled by scale_factor (for precision)
    pub(crate) upper_scaled: i64,
    /// Number of discrete steps in the range
    pub(crate) steps: u32,
    /// Scale factor for decimal precision (e.g., 1_000_000 for 6 decimal places)
    pub(crate) scale_factor: i64,
}

impl GeneBounds {
    /// Create bounds with decimal precision using fixed-point arithmetic
    /// precision: number of decimal places (e.g., 6 for microsecond precision)
    #[instrument(level = "debug", fields(lower = lower, upper = upper, steps = steps, precision = precision))]
    pub fn decimal(
        lower: f64,
        upper: f64,
        steps: u32,
        precision: u8,
    ) -> Result<Self, GeneBoundError> {
        if lower >= upper {
            return Err(GeneBoundError::invalid_bound(lower, upper));
        }
        if steps == 0 {
            return Err(GeneBoundError::zero_steps());
        }

        let scale_factor = 10_i64.pow(precision as u32);
        let lower_scaled = (lower * scale_factor as f64).round() as i64;
        let upper_scaled = (upper * scale_factor as f64).round() as i64;

        Ok(Self {
            lower_scaled,
            upper_scaled,
            steps,
            scale_factor,
        })
    }

    /// Create integer-only bounds for backward compatibility
    #[instrument(level = "debug", fields(lower = lower, upper = upper, steps = steps))]
    pub fn integer(lower: i32, upper: i32, steps: u32) -> Result<Self, GeneBoundError> {
        if lower > upper {
            return Err(GeneBoundError::invalid_bound(lower as f64, upper as f64));
        }

        let steps = i32::try_from(steps).map_err(|_| GeneBoundError::steps_overflow(steps))?;

        // Convert to new format: scale factor of 1 means integer precision
        Ok(Self {
            lower_scaled: lower as i64,
            upper_scaled: upper as i64,
            steps: steps as u32,
            scale_factor: 1,
        })
    }

    /// Convert gene to decimal value
    pub fn to_f64(&self, gene: Gene) -> f64 {
        let range = self.upper_scaled - self.lower_scaled;
        let scaled_value = self.lower_scaled + (gene * range) / (self.steps - 1) as i64;
        scaled_value as f64 / self.scale_factor as f64
    }

    /// Generate a random gene (0 to steps-1)
    #[instrument(level = "debug", skip(rng), fields(steps = self.steps))]
    pub(crate) fn random(&self, rng: &mut ThreadRng) -> Gene {
        rng.random_range(0..self.steps as i64)
    }

    pub(crate) fn steps(&self) -> i32 {
        self.steps as i32
    }

    /// Convert a [0,1] sample to a gene index (0 to steps-1)
    pub(crate) fn from_sample(&self, sample: f64) -> Gene {
        // Map [0,1] sample to discrete gene index [0, steps-1]
        (sample * (self.steps - 1) as f64).round() as Gene
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_sample() {
        let bounds = GeneBounds::decimal(0.0, 10.0, 100, 3).unwrap();

        // Test exact boundaries
        assert_eq!(bounds.from_sample(0.0), 0); // Minimum sample
        assert_eq!(bounds.from_sample(1.0), 99); // Maximum sample (steps-1)

        // Test middle point
        assert_eq!(bounds.from_sample(0.5), 50); // Middle sample → 0.5 * 99 = 49.5 → rounds to 50

        // Test quarter points
        assert_eq!(bounds.from_sample(0.25), 25); // 0.25 * 99 = 24.75 → rounds to 25
        assert_eq!(bounds.from_sample(0.75), 74); // 0.75 * 99 = 74.25 → rounds to 74

        // Test very small positive values (near 0 but not 0)
        assert_eq!(bounds.from_sample(0.001), 0); // 0.001 * 99 = 0.099 → rounds to 0
        assert_eq!(bounds.from_sample(0.01), 1); // 0.01 * 99 = 0.99 → rounds to 1

        // Test very close to 1 (but not 1)
        assert_eq!(bounds.from_sample(0.999), 99); // 0.999 * 99 = 98.901 → rounds to 99
        assert_eq!(bounds.from_sample(0.99), 98); // 0.99 * 99 = 98.01 → rounds to 98

        // Test Latin Hypercube typical values
        // For n_samples=4: [0.125, 0.375, 0.625, 0.875]
        assert_eq!(bounds.from_sample(0.125), 12); // 0.125 * 99 = 12.375 → rounds to 12
        assert_eq!(bounds.from_sample(0.375), 37); // 0.375 * 99 = 37.125 → rounds to 37
        assert_eq!(bounds.from_sample(0.625), 62); // 0.625 * 99 = 61.875 → rounds to 62
        assert_eq!(bounds.from_sample(0.875), 87); // 0.875 * 99 = 86.625 → rounds to 87

        // Test edge case with single step
        let single_step = GeneBounds::decimal(0.0, 1.0, 1, 3).unwrap();
        assert_eq!(single_step.from_sample(0.0), 0); // Only possible gene
        assert_eq!(single_step.from_sample(0.5), 0); // 0.5 * 0 = 0
        assert_eq!(single_step.from_sample(1.0), 0); // 1.0 * 0 = 0

        // Test with different step count
        let bounds_10 = GeneBounds::decimal(0.0, 5.0, 10, 2).unwrap();
        assert_eq!(bounds_10.from_sample(0.0), 0); // Min
        assert_eq!(bounds_10.from_sample(1.0), 9); // Max (steps-1)
        assert_eq!(bounds_10.from_sample(0.5), 5); // 0.5 * 9 = 4.5 → rounds to 5
    }

    #[test]
    fn test_integer_bounds_backward_compatibility() {
        let bounds = GeneBounds::integer(1, 10, 10).unwrap();
        assert_eq!(bounds.lower_scaled, 1);
        assert_eq!(bounds.upper_scaled, 10);
        assert_eq!(bounds.steps, 10);
        assert_eq!(bounds.scale_factor, 1);
        assert_eq!(bounds.steps(), 10);
    }

    #[test]
    fn test_decimal_bounds() {
        let bounds = GeneBounds::decimal(0.23, 0.5, 500, 6).unwrap();
        assert_eq!(bounds.lower_scaled, 230000); // 0.23 * 1_000_000
        assert_eq!(bounds.upper_scaled, 500000); // 0.5 * 1_000_000
        assert_eq!(bounds.steps, 500);
        assert_eq!(bounds.scale_factor, 1_000_000);
    }

    #[test]
    fn test_invalid_bounds() {
        // Test lower >= upper
        assert!(GeneBounds::decimal(0.5, 0.23, 500, 6).is_err());
        assert!(GeneBounds::decimal(0.5, 0.5, 500, 6).is_err());

        // Test zero steps
        assert!(GeneBounds::decimal(0.23, 0.5, 0, 6).is_err());
    }
}
