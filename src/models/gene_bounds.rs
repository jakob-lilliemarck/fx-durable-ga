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

    /// Generate a random gene (0 to steps-1)
    #[instrument(level = "debug", skip(rng), fields(steps = self.steps))]
    pub fn random(&self, rng: &mut ThreadRng) -> Gene {
        rng.random_range(0..self.steps as i64)
    }

    /// Convert gene to decimal value
    pub fn decode_gene(&self, gene: Gene) -> f64 {
        let range = self.upper_scaled - self.lower_scaled;
        let scaled_value = self.lower_scaled + (gene * range) / (self.steps - 1) as i64;
        scaled_value as f64 / self.scale_factor as f64
    }

    pub fn steps(&self) -> i32 {
        self.steps as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
