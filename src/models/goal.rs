use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Defines the optimization objective and termination criteria for a genetic algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FitnessGoal {
    /// Minimize fitness values, stopping when fitness drops to or below the threshold.
    Minimize { threshold: f64 },
    /// Maximize fitness values, stopping when fitness reaches or exceeds the threshold.
    Maximize { threshold: f64 },
}

#[derive(Debug, thiserror::Error)]
#[error("fitness goal threshold must be between 0.0 and 1.0, got {0}")]
pub struct ThresholdOutOfRange(f64);

impl FitnessGoal {
    /// Creates a minimization goal with the given threshold.
    pub fn minimize(threshold: f64) -> Result<Self, ThresholdOutOfRange> {
        let threshold = Self::validate(threshold)?;

        Ok(Self::Minimize { threshold })
    }

    /// Creates a maximization goal with the given threshold.
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

    /// Calculates optimization progress from 0.0 (no progress) to 1.0 (goal reached).
    /// Returns 0.0 if best_fitness is None.
    #[instrument(level = "debug", skip(self), fields(goal = ?self, best_fitness = ?best_fitness))]
    pub(crate) fn calculate_progress(&self, best_fitness: Option<f64>) -> f64 {
        let best_fitness = match best_fitness {
            Some(fitness) => fitness,
            None => return 0.0, // No fitness data = no progress
        };

        match self {
            FitnessGoal::Maximize { threshold } => {
                // Progress = current_fitness / threshold, clamped to [0.0, 1.0]
                (best_fitness / threshold).min(1.0).max(0.0)
            }
            FitnessGoal::Minimize { threshold } => {
                // Progress = (1.0 - current_fitness) / (1.0 - threshold)
                if *threshold >= 1.0 {
                    return 1.0; // Edge case: threshold is 1.0
                }
                ((1.0 - best_fitness) / (1.0 - threshold)).min(1.0).max(0.0)
            }
        }
    }

    /// Validates that the threshold is within the valid range [0.0, 1.0].
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

    #[test]
    fn test_progress_calculation_maximize() {
        let goal = FitnessGoal::maximize(0.8).unwrap();

        // No fitness data = no progress
        assert_eq!(goal.calculate_progress(None), 0.0);

        // Progress from 0.0 to threshold
        assert_eq!(goal.calculate_progress(Some(0.0)), 0.0); // 0/0.8 = 0.0
        assert_eq!(goal.calculate_progress(Some(0.4)), 0.5); // 0.4/0.8 = 0.5
        assert_eq!(goal.calculate_progress(Some(0.8)), 1.0); // 0.8/0.8 = 1.0

        // Exceeding threshold should clamp to 1.0
        assert_eq!(goal.calculate_progress(Some(1.0)), 1.0); // Clamped

        // Negative values should clamp to 0.0
        assert_eq!(goal.calculate_progress(Some(-0.1)), 0.0); // Clamped
    }

    #[test]
    fn test_progress_calculation_minimize() {
        let goal = FitnessGoal::minimize(0.2).unwrap();

        // No fitness data = no progress
        assert_eq!(goal.calculate_progress(None), 0.0);

        // Progress formula: (1.0 - current) / (1.0 - threshold)
        // With threshold 0.2: (1.0 - current) / 0.8
        assert_eq!(goal.calculate_progress(Some(1.0)), 0.0); // (1.0-1.0)/0.8 = 0.0
        assert_eq!(goal.calculate_progress(Some(0.6)), 0.5); // (1.0-0.6)/0.8 = 0.5
        assert_eq!(goal.calculate_progress(Some(0.2)), 1.0); // (1.0-0.2)/0.8 = 1.0

        // Better than threshold should clamp to 1.0
        assert_eq!(goal.calculate_progress(Some(0.1)), 1.0); // Clamped

        // Worse than 1.0 should clamp to 0.0
        assert_eq!(goal.calculate_progress(Some(1.1)), 0.0); // Clamped
    }

    #[test]
    fn test_progress_calculation_edge_cases() {
        // Edge case: minimize with threshold 1.0
        let min_goal_edge = FitnessGoal::minimize(1.0).unwrap();
        assert_eq!(min_goal_edge.calculate_progress(Some(0.5)), 1.0); // Special case
        assert_eq!(min_goal_edge.calculate_progress(Some(1.0)), 1.0); // Special case

        // Edge case: maximize with threshold very close to 0
        let max_goal_edge = FitnessGoal::maximize(0.001).unwrap();
        assert_eq!(max_goal_edge.calculate_progress(Some(0.0005)), 0.5); // 0.0005/0.001 = 0.5
    }
}
