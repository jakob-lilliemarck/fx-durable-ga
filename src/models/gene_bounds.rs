use super::Gene;
use rand::{Rng, rngs::ThreadRng};
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Errors that can occur when creating gene bounds.
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
    /// Creates a steps overflow error.
    pub(crate) fn steps_overflow(steps: u32) -> Self {
        Self::StepsOverflow {
            steps,
            max: i32::MAX,
        }
    }

    /// Creates an invalid bound error.
    pub(crate) fn invalid_bound(lower: f64, upper: f64) -> Self {
        Self::InvalidBound { lower, upper }
    }

    /// Creates a zero steps error.
    pub(crate) fn zero_steps() -> Self {
        Self::ZeroSteps
    }
}

/// Defines the bounds and discretization for a single gene in the search space.
/// Uses fixed-point arithmetic for precise decimal handling.
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
    /// Creates bounds with decimal precision using fixed-point arithmetic.
    /// Precision specifies the number of decimal places (e.g., 6 for microsecond precision).
    #[instrument(level = "debug", fields(lower = lower, upper = upper, steps = steps, precision = precision))]
    pub fn decimal(
        lower: f64,
        upper: f64,
        steps: u32,
        precision: u8,
    ) -> Result<Self, GeneBoundError> {
        Self::validate_bounds(lower, upper, steps)?;

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

    /// Creates integer-only bounds with no fractional precision.
    #[instrument(level = "debug", fields(lower = lower, upper = upper, steps = steps))]
    pub fn integer(lower: i32, upper: i32, steps: u32) -> Result<Self, GeneBoundError> {
        Self::validate_bounds(lower, upper, steps)?;

        // Convert to new format: scale factor of 1 means integer precision
        Ok(Self {
            lower_scaled: lower as i64,
            upper_scaled: upper as i64,
            steps: steps as u32,
            scale_factor: 1,
        })
    }

    /// Validates that bounds are sensible and within system limits.
    fn validate_bounds<T>(lower: T, upper: T, steps: u32) -> Result<(), GeneBoundError>
    where
        T: PartialOrd + Copy + Into<f64>,
    {
        if lower > upper {
            return Err(GeneBoundError::invalid_bound(lower.into(), upper.into()));
        }

        if steps == 0 {
            return Err(GeneBoundError::zero_steps());
        }

        if steps > i32::MAX as u32 {
            return Err(GeneBoundError::steps_overflow(steps));
        }

        if lower == upper && steps > 1 {
            return Err(GeneBoundError::invalid_bound(lower.into(), upper.into()));
        }

        Ok(())
    }

    /// Converts a discrete gene value to its corresponding decimal value in the real range.
    pub fn to_f64(&self, gene: Gene) -> f64 {
        let range = self.upper_scaled - self.lower_scaled;
        let scaled_value = self.lower_scaled + (gene * range) / (self.steps - 1) as i64;
        scaled_value as f64 / self.scale_factor as f64
    }

    /// Generates a random gene value within the discrete range [0, steps-1].
    #[instrument(level = "debug", skip(rng), fields(steps = self.steps))]
    pub(crate) fn random(&self, rng: &mut ThreadRng) -> Gene {
        rng.random_range(0..self.steps as i64)
    }

    /// Returns the number of discrete steps as an i32.
    pub(crate) fn steps(&self) -> i32 {
        self.steps as i32
    }

    /// Converts a normalized [0,1] sample to a discrete gene index [0, steps-1].
    pub fn from_sample(&self, sample: f64) -> Gene {
        // Map [0,1] sample to discrete gene index [0, steps-1]
        (sample * (self.steps - 1) as f64).round() as Gene
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_ordering_validation() {
        // lower > upper should fail for both types
        assert!(GeneBounds::decimal(0.5, 0.23, 10, 6).is_err());
        assert!(GeneBounds::integer(10, 5, 10).is_err());
    }

    #[test]
    fn test_zero_steps_validation() {
        // zero steps should fail for both types
        assert!(GeneBounds::decimal(0.0, 1.0, 0, 6).is_err());
        assert!(GeneBounds::integer(0, 10, 0).is_err());
    }

    #[test]
    fn test_steps_overflow_validation() {
        // steps overflow should fail for both types
        assert!(GeneBounds::decimal(0.0, 1.0, u32::MAX, 6).is_err());
        assert!(GeneBounds::integer(0, 10, u32::MAX).is_err());
    }

    #[test]
    fn test_equal_bounds_validation() {
        // equal bounds with steps > 1 should fail
        assert!(GeneBounds::decimal(0.5, 0.5, 2, 6).is_err());
        assert!(GeneBounds::integer(5, 5, 2).is_err());

        // equal bounds with steps = 1 should succeed
        assert!(GeneBounds::decimal(0.5, 0.5, 1, 6).is_ok());
        assert!(GeneBounds::integer(5, 5, 1).is_ok());
    }

    #[test]
    fn test_from_sample_boundaries() {
        let bounds = GeneBounds::decimal(0.0, 10.0, 100, 3).unwrap();
        assert_eq!(bounds.from_sample(0.0), 0);
        assert_eq!(bounds.from_sample(1.0), 99);
        assert_eq!(bounds.from_sample(0.5), 50);
    }

    #[test]
    fn test_from_sample_single_step() {
        let bounds = GeneBounds::decimal(0.0, 1.0, 1, 3).unwrap();
        assert_eq!(bounds.from_sample(0.0), 0);
        assert_eq!(bounds.from_sample(1.0), 0);
    }

    #[test]
    fn test_from_sample_different_step_counts() {
        let bounds = GeneBounds::decimal(0.0, 5.0, 10, 2).unwrap();
        assert_eq!(bounds.from_sample(0.0), 0);
        assert_eq!(bounds.from_sample(1.0), 9);
        assert_eq!(bounds.from_sample(0.5), 5);
    }

    #[test]
    fn test_bounds_inclusion() {
        // Test that both lower and upper bounds are included in the range
        let bounds = GeneBounds::decimal(0.0, 1.0, 2, 3).unwrap();

        // With 2 steps, we should get exactly the lower and upper bounds
        assert_eq!(bounds.to_f64(0), 0.0); // Lower bound included
        assert_eq!(bounds.to_f64(1), 1.0); // Upper bound included

        // Verify from_sample maps correctly to these discrete values
        assert_eq!(bounds.from_sample(0.0), 0); // Maps to lower bound
        assert_eq!(bounds.from_sample(1.0), 1); // Maps to upper bound

        // Test with more steps to show even distribution
        let bounds_5 = GeneBounds::decimal(0.0, 4.0, 5, 1).unwrap();
        assert_eq!(bounds_5.to_f64(0), 0.0); // 0.0
        assert_eq!(bounds_5.to_f64(1), 1.0); // 1.0
        assert_eq!(bounds_5.to_f64(2), 2.0); // 2.0
        assert_eq!(bounds_5.to_f64(3), 3.0); // 3.0
        assert_eq!(bounds_5.to_f64(4), 4.0); // 4.0 (upper bound included)
    }

    #[test]
    fn test_integer_bounds_internal_representation() {
        let bounds = GeneBounds::integer(1, 10, 10).unwrap();
        assert_eq!(bounds.lower_scaled, 1);
        assert_eq!(bounds.upper_scaled, 10);
        assert_eq!(bounds.steps, 10);
        assert_eq!(bounds.scale_factor, 1);
        assert_eq!(bounds.steps(), 10);
    }

    #[test]
    fn test_decimal_bounds_internal_representation() {
        let bounds = GeneBounds::decimal(0.23, 0.5, 500, 6).unwrap();
        assert_eq!(bounds.lower_scaled, 230000); // 0.23 * 1_000_000
        assert_eq!(bounds.upper_scaled, 500000); // 0.5 * 1_000_000
        assert_eq!(bounds.steps, 500);
        assert_eq!(bounds.scale_factor, 1_000_000);
    }

    #[test]
    fn test_to_f64_conversion() {
        // Test decimal conversion
        let bounds = GeneBounds::decimal(0.0, 10.0, 11, 1).unwrap(); // 11 steps: 0,1,2,...,10

        assert_eq!(bounds.to_f64(0), 0.0); // First step
        assert_eq!(bounds.to_f64(5), 5.0); // Middle step
        assert_eq!(bounds.to_f64(10), 10.0); // Last step

        // Test with higher precision
        let precise_bounds = GeneBounds::decimal(0.0, 1.0, 101, 3).unwrap(); // 0.000 to 1.000
        assert!((precise_bounds.to_f64(0) - 0.0).abs() < 0.001);
        assert!((precise_bounds.to_f64(50) - 0.5).abs() < 0.01);
        assert!((precise_bounds.to_f64(100) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_random_generates_different_values() {
        let bounds = GeneBounds::integer(0, 1000, 100).unwrap(); // Large range
        let mut rng = rand::rng();

        let gene1 = bounds.random(&mut rng);
        let gene2 = bounds.random(&mut rng);

        // With 100 possible values, random genes are extremely unlikely to be equal
        assert_ne!(gene1, gene2);

        // Verify bounds
        assert!(gene1 >= 0 && gene1 < 100);
        assert!(gene2 >= 0 && gene2 < 100);
    }
}
