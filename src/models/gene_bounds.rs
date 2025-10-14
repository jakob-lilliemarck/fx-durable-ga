use super::Gene;
use rand::{Rng, rngs::ThreadRng};
use serde::{Deserialize, Serialize};
use tracing::instrument;

#[derive(Debug, thiserror::Error)]
pub enum GeneBoundError {
    #[error(
        "InvalidBounds: lower bound must be smaller than upper. lower = {lower}, upper={upper}"
    )]
    InvalidBound { lower: i32, upper: i32 },
    #[error("DivisorOverflow: divisor is too large. divisor={divisor}, max={max}")]
    DivisorOverflow { divisor: u32, max: i32 },
}

impl GeneBoundError {
    pub(crate) fn divisor_overflow(divisor: u32) -> Self {
        Self::DivisorOverflow {
            divisor,
            max: i32::MAX,
        }
    }

    pub(crate) fn invalid_bound(lower: i32, upper: i32) -> Self {
        Self::InvalidBound { lower, upper }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct GeneBounds {
    pub(crate) lower: i32,
    pub(crate) upper: i32,
    pub(crate) divisor: i32,
}

impl GeneBounds {
    #[instrument(level = "debug", fields(lower = lower, upper = upper, divisor = divisor))]
    pub fn new(lower: i32, upper: i32, divisor: u32) -> Result<Self, GeneBoundError> {
        if lower > upper {
            return Err(GeneBoundError::invalid_bound(lower, upper));
        };

        let divisor =
            i32::try_from(divisor).map_err(|_| GeneBoundError::divisor_overflow(divisor))?;

        Ok(Self {
            lower,
            upper,
            divisor,
        })
    }

    #[instrument(level = "debug", skip(rng), fields(lower = self.lower, upper = self.upper, divisor = self.divisor))]
    pub fn random(&self, rng: &mut ThreadRng) -> Gene {
        rng.random_range(0..self.divisor as i64)
    }
}
