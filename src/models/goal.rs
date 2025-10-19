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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_thresholds() {
        // Test minimize with invalid threshold
        assert!(FitnessGoal::minimize(-0.1).is_err());
        assert!(FitnessGoal::minimize(1.5).is_err());

        // Test maximize with invalid threshold
        assert!(FitnessGoal::maximize(-0.1).is_err());
        assert!(FitnessGoal::maximize(1.5).is_err());
    }

    #[test]
    fn test_is_reached_minimize() {
        let goal = FitnessGoal::minimize(0.5).unwrap();

        assert!(goal.is_reached(0.3)); // Below threshold
        assert!(goal.is_reached(0.5)); // At threshold
        assert!(!goal.is_reached(0.7)); // Above threshold
    }

    #[test]
    fn test_is_reached_maximize() {
        let goal = FitnessGoal::maximize(0.5).unwrap();

        assert!(!goal.is_reached(0.3)); // Below threshold
        assert!(goal.is_reached(0.5)); // At threshold
        assert!(goal.is_reached(0.7)); // Above threshold
    }

    #[test]
    fn test_boundary_values() {
        let min_goal = FitnessGoal::minimize(0.0).unwrap();
        let max_goal = FitnessGoal::maximize(1.0).unwrap();

        // Test that 0.0 threshold for minimize accepts everything
        assert!(min_goal.is_reached(0.0));
        assert!(!min_goal.is_reached(0.1)); // Even tiny positive values fail

        // Test that 1.0 threshold for maximize accepts only perfect scores
        assert!(max_goal.is_reached(1.0));
        assert!(!max_goal.is_reached(0.9)); // Even near-perfect scores fail
    }
}
