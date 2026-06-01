use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Optimization termination criteria - when to stop the genetic algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FitnessGoal {
    /// Stop when fitness ≤ threshold (lower values are better).
    Minimize { threshold: f64 },
    /// Stop when fitness ≥ threshold (higher values are better).
    Maximize { threshold: f64 },
}

/// Error when threshold is NaN or infinite.
#[derive(Debug, thiserror::Error)]
#[error("fitness goal threshold must be finite (not NaN or infinite), got {0}")]
pub struct ThresholdOutOfRange(f64);

impl FitnessGoal {
    /// Stop when fitness ≤ threshold.
    pub fn minimize(threshold: f64) -> Result<Self, ThresholdOutOfRange> {
        let threshold = Self::validate(threshold)?;

        Ok(Self::Minimize { threshold })
    }

    /// Stop when fitness ≥ threshold.
    pub fn maximize(threshold: f64) -> Result<Self, ThresholdOutOfRange> {
        let threshold = Self::validate(threshold)?;

        Ok(Self::Maximize { threshold })
    }

    /// Checks if the given fitness value has reached the goal threshold.
    #[instrument(level = "debug", skip(self), fields(goal = ?self, fitness = fitness))]
    pub(crate) fn is_reached(&self, fitness: f64) -> bool {
        match self {
            FitnessGoal::Minimize { threshold } => fitness <= *threshold,
            FitnessGoal::Maximize { threshold } => fitness >= *threshold,
        }
    }

    /// Determines if candidate_fitness is better than current_best_fitness for this goal.
    pub(crate) fn is_better(&self, candidate_fitness: f64, current_best_fitness: f64) -> bool {
        match self {
            FitnessGoal::Maximize { .. } => candidate_fitness > current_best_fitness,
            FitnessGoal::Minimize { .. } => candidate_fitness < current_best_fitness,
        }
    }

    /// Validates that the threshold is a valid f64 value (not NaN or infinite).
    fn validate(threshold: f64) -> Result<f64, ThresholdOutOfRange> {
        if !threshold.is_finite() {
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
        assert!(FitnessGoal::minimize(f64::NAN).is_err());
        assert!(FitnessGoal::minimize(f64::INFINITY).is_err());
        assert!(FitnessGoal::minimize(f64::NEG_INFINITY).is_err());

        assert!(FitnessGoal::maximize(f64::NAN).is_err());
        assert!(FitnessGoal::maximize(f64::INFINITY).is_err());
        assert!(FitnessGoal::maximize(f64::NEG_INFINITY).is_err());

        assert!(FitnessGoal::minimize(-100.0).is_ok());
        assert!(FitnessGoal::minimize(1000.0).is_ok());
        assert!(FitnessGoal::maximize(-50.0).is_ok());
        assert!(FitnessGoal::maximize(500.0).is_ok());
    }

    #[test]
    fn test_is_reached_minimize() {
        let goal = FitnessGoal::minimize(0.5).unwrap();

        assert!(goal.is_reached(0.3));
        assert!(goal.is_reached(0.5));
        assert!(!goal.is_reached(0.7));
    }

    #[test]
    fn test_is_reached_maximize() {
        let goal = FitnessGoal::maximize(0.5).unwrap();

        assert!(!goal.is_reached(0.3));
        assert!(goal.is_reached(0.5));
        assert!(goal.is_reached(0.7));
    }

    #[test]
    fn test_boundary_values() {
        let min_goal = FitnessGoal::minimize(0.0).unwrap();
        let max_goal = FitnessGoal::maximize(100.0).unwrap();

        assert!(min_goal.is_reached(0.0));
        assert!(min_goal.is_reached(-0.1));
        assert!(!min_goal.is_reached(0.1));

        assert!(max_goal.is_reached(100.0));
        assert!(max_goal.is_reached(150.0));
        assert!(!max_goal.is_reached(99.9));
    }
}
