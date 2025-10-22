//! # Fitness Goal Configuration
//!
//! Defines when genetic algorithm optimization should stop based on achieved fitness values.
//!
//! ## Usage
//!
//! ```rust
//! use fx_durable_ga::models::FitnessGoal;
//!
//! // Stop when loss ≤ 0.05 (lower is better)
//! let goal = FitnessGoal::minimize(0.05)?;
//!
//! // Stop when accuracy ≥ 95% (higher is better)
//! let goal = FitnessGoal::maximize(0.95)?;
//!
//! // Works with any values: costs, profits, scores, etc.
//! let goal = FitnessGoal::minimize(1000.0)?; // Stop when cost ≤ $1000
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Optimization termination criteria - when to stop the genetic algorithm.
///
/// ```rust
/// use fx_durable_ga::models::FitnessGoal;
///
/// let goal = FitnessGoal::minimize(0.05)?;  // Stop when loss ≤ 0.05
/// let goal = FitnessGoal::maximize(0.95)?;  // Stop when accuracy ≥ 95%
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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

    /// Returns the better fitness value according to this goal.
    pub(crate) fn best_fitness<'a>(
        &'a self,
        min_fitness: &'a Option<f64>,
        max_fitness: &'a Option<f64>,
    ) -> &'a Option<f64> {
        match self {
            FitnessGoal::Maximize { .. } => max_fitness,
            FitnessGoal::Minimize { .. } => min_fitness,
        }
    }

    /// FIXME: Progress calculation needs to be redesigned for unbounded fitness values.
    /// For now, always returns 0.0 which means decay strategies will use constant values.
    /// Returns 0.0 if best_fitness is None.
    #[instrument(level = "debug", skip(self), fields(goal = ?self))]
    pub(crate) fn calculate_progress(&self, _: Option<f64>) -> f64 {
        // FIXME: Implement proper progress calculation for unbounded fitness
        // Options: generation-based progress, improvement-rate based, or user-defined decay functions
        0.0
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
        // Test minimize with invalid threshold (NaN/infinite)
        assert!(FitnessGoal::minimize(f64::NAN).is_err());
        assert!(FitnessGoal::minimize(f64::INFINITY).is_err());
        assert!(FitnessGoal::minimize(f64::NEG_INFINITY).is_err());

        // Test maximize with invalid threshold (NaN/infinite)
        assert!(FitnessGoal::maximize(f64::NAN).is_err());
        assert!(FitnessGoal::maximize(f64::INFINITY).is_err());
        assert!(FitnessGoal::maximize(f64::NEG_INFINITY).is_err());

        // These should now be valid (previously invalid bounded values)
        assert!(FitnessGoal::minimize(-100.0).is_ok());
        assert!(FitnessGoal::minimize(1000.0).is_ok());
        assert!(FitnessGoal::maximize(-50.0).is_ok());
        assert!(FitnessGoal::maximize(500.0).is_ok());
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
        let max_goal = FitnessGoal::maximize(100.0).unwrap();

        // Test that 0.0 threshold for minimize accepts only zero or negative
        assert!(min_goal.is_reached(0.0));
        assert!(min_goal.is_reached(-0.1)); // Negative values pass
        assert!(!min_goal.is_reached(0.1)); // Positive values fail

        // Test that 100.0 threshold for maximize accepts 100.0 or higher
        assert!(max_goal.is_reached(100.0));
        assert!(max_goal.is_reached(150.0)); // Higher values pass
        assert!(!max_goal.is_reached(99.9)); // Lower values fail
    }

    #[test]
    fn test_progress_calculation_maximize() {
        let goal = FitnessGoal::maximize(0.8).unwrap();

        // FIXME: Progress calculation always returns 0.0 for now
        assert_eq!(goal.calculate_progress(None), 0.0);
        assert_eq!(goal.calculate_progress(Some(0.0)), 0.0);
        assert_eq!(goal.calculate_progress(Some(0.4)), 0.0);
        assert_eq!(goal.calculate_progress(Some(0.8)), 0.0);
        assert_eq!(goal.calculate_progress(Some(1.0)), 0.0);
        assert_eq!(goal.calculate_progress(Some(-0.1)), 0.0);
    }

    #[test]
    fn test_progress_calculation_minimize() {
        let goal = FitnessGoal::minimize(0.2).unwrap();

        // FIXME: Progress calculation always returns 0.0 for now
        assert_eq!(goal.calculate_progress(None), 0.0);
        assert_eq!(goal.calculate_progress(Some(1.0)), 0.0);
        assert_eq!(goal.calculate_progress(Some(0.6)), 0.0);
        assert_eq!(goal.calculate_progress(Some(0.2)), 0.0);
        assert_eq!(goal.calculate_progress(Some(0.1)), 0.0);
        assert_eq!(goal.calculate_progress(Some(1.1)), 0.0);
    }

    #[test]
    fn test_progress_calculation_edge_cases() {
        // FIXME: Progress calculation always returns 0.0 for now
        let min_goal_edge = FitnessGoal::minimize(1.0).unwrap();
        assert_eq!(min_goal_edge.calculate_progress(Some(0.5)), 0.0);
        assert_eq!(min_goal_edge.calculate_progress(Some(1.0)), 0.0);

        let max_goal_edge = FitnessGoal::maximize(0.001).unwrap();
        assert_eq!(max_goal_edge.calculate_progress(Some(0.0005)), 0.0);
    }
}
