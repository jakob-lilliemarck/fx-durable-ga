//! # Fitness Goal Configuration
//!
//! This module provides types for defining optimization objectives and termination criteria
//! in genetic algorithms. The primary type, [`FitnessGoal`], determines when the optimization
//! process should stop based on achieved fitness values.
//!
//! ## Core Concepts
//!
//! **Fitness Values**: All fitness scores in this system are normalized to the range [0.0, 1.0].
//! Your fitness function should return values within this range for optimal behavior.
//!
//! **Termination**: The genetic algorithm continues evolving generations until any individual
//! in the population achieves the target fitness threshold, at which point optimization stops.
//!
//! ## Choosing the Right Goal Type
//!
//! ### Minimization Goals
//!
//! Use [`FitnessGoal::Minimize`] when lower fitness values represent better solutions:
//!
//! - **Error minimization**: Fitness = error_rate (0.0 = perfect, 1.0 = worst)
//! - **Cost optimization**: Fitness = cost / max_cost (0.0 = free, 1.0 = maximum cost)
//! - **Distance problems**: Fitness = distance / max_distance (0.0 = perfect match, 1.0 = furthest)
//!
//! ### Maximization Goals  
//!
//! Use [`FitnessGoal::Maximize`] when higher fitness values represent better solutions:
//!
//! - **Accuracy optimization**: Fitness = accuracy (0.0 = 0% accurate, 1.0 = 100% accurate)
//! - **Performance optimization**: Fitness = performance / max_performance (1.0 = optimal)
//! - **Quality metrics**: Fitness = quality_score (1.0 = highest quality)
//!
//! ## Threshold Selection Guidelines
//!
//! ### For Minimization
//! - **Loose**: 0.1 (stop at 10% of maximum error/cost)
//! - **Moderate**: 0.05 (stop at 5% of maximum error/cost)
//! - **Strict**: 0.01 (stop at 1% of maximum error/cost)
//! - **Very strict**: 0.001 (stop at 0.1% of maximum error/cost)
//!
//! ### For Maximization
//! - **Loose**: 0.8 (stop at 80% accuracy/performance)
//! - **Moderate**: 0.9 (stop at 90% accuracy/performance)
//! - **Strict**: 0.95 (stop at 95% accuracy/performance)
//! - **Very strict**: 0.99+ (stop at 99%+ accuracy/performance)
//!
//! ## Configuration Examples
//!
//! ### Machine Learning Model Training
//!
//! ```rust
//! use fx_durable_ga::models::FitnessGoal;
//!
//! // Stop when training error drops below 2%
//! let training_goal = FitnessGoal::minimize(0.02)?;
//!
//! // Stop when validation accuracy reaches 95%
//! let validation_goal = FitnessGoal::maximize(0.95)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Engineering Optimization
//!
//! ```rust
//! use fx_durable_ga::models::FitnessGoal;
//!
//! // Stop when structural weight is reduced to 15% of baseline
//! let weight_goal = FitnessGoal::minimize(0.15)?;
//!
//! // Stop when efficiency reaches 90% of theoretical maximum
//! let efficiency_goal = FitnessGoal::maximize(0.90)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Financial Optimization
//!
//! ```rust
//! use fx_durable_ga::models::FitnessGoal;
//!
//! // Stop when portfolio risk is reduced to 5% of maximum
//! let risk_goal = FitnessGoal::minimize(0.05)?;
//!
//! // Stop when expected return reaches 85% of theoretical maximum
//! let return_goal = FitnessGoal::maximize(0.85)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Integration with Optimization Requests
//!
//! Fitness goals are typically used when creating optimization requests:
//!
//! ```rust,no_run
//! # use fx_durable_ga::models::*;
//! # use fx_durable_ga::services::optimization::Service;
//! # async fn example(service: &Service) -> Result<(), Box<dyn std::error::Error>> {
//! service.new_optimization_request(
//!     "MyProblemType",
//!     12345, // type hash
//!     FitnessGoal::maximize(0.95)?, // Stop at 95% accuracy
//!     Schedule::generational(100, 10),
//!     Selector::tournament(3, 50),
//!     Mutagen::new(
//!         Temperature::constant(0.5)?,
//!         MutationRate::constant(0.1)?,
//!     ),
//!     Crossover::uniform(0.5)?,
//!     Distribution::random(50),
//! ).await?;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Defines the optimization objective and termination criteria for a genetic algorithm.
///
/// `FitnessGoal` specifies whether to minimize or maximize fitness values, along with the
/// threshold that determines when optimization is considered complete. All thresholds must
/// be between 0.0 and 1.0.
///
/// # When to Use Each Variant
///
/// - **Minimize**: Use when lower fitness scores represent better solutions (e.g., error minimization, cost reduction)
/// - **Maximize**: Use when higher fitness scores represent better solutions (e.g., accuracy, profit)
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::FitnessGoal;
///
/// // For error minimization - stop when error drops below 5%
/// let error_goal = FitnessGoal::minimize(0.05)?;
///
/// // For accuracy maximization - stop when accuracy reaches 95%
/// let accuracy_goal = FitnessGoal::maximize(0.95)?;
///
/// // High-precision optimization - stop at 99.9% accuracy
/// let precision_goal = FitnessGoal::maximize(0.999)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FitnessGoal {
    /// Minimize fitness values, stopping when fitness drops to or below the threshold.
    ///
    /// Use this variant when lower fitness scores represent better solutions.
    /// The algorithm terminates when any individual achieves a fitness value ≤ threshold.
    ///
    /// Common use cases:
    /// - Error rate minimization (threshold: 0.01 for 1% error)
    /// - Cost optimization (threshold: 0.1 for 10% of baseline cost)
    /// - Distance minimization (threshold: 0.05 for 5% of maximum distance)
    Minimize { threshold: f64 },
    /// Maximize fitness values, stopping when fitness reaches or exceeds the threshold.
    ///
    /// Use this variant when higher fitness scores represent better solutions.
    /// The algorithm terminates when any individual achieves a fitness value ≥ threshold.
    ///
    /// Common use cases:
    /// - Accuracy maximization (threshold: 0.95 for 95% accuracy)
    /// - Profit optimization (threshold: 0.8 for 80% of maximum profit)
    /// - Performance optimization (threshold: 0.99 for 99% efficiency)
    Maximize { threshold: f64 },
}

/// Error returned when a fitness goal threshold is outside the valid range [0.0, 1.0].
///
/// This error occurs when attempting to create a `FitnessGoal` with a threshold
/// value that is negative or greater than 1.0. All fitness values in this system
/// are normalized to the range [0.0, 1.0].
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::FitnessGoal;
///
/// // These will return ThresholdOutOfRange errors:
/// assert!(FitnessGoal::maximize(-0.1).is_err());
/// assert!(FitnessGoal::minimize(1.5).is_err());
///
/// // These are valid:
/// assert!(FitnessGoal::maximize(0.0).is_ok());
/// assert!(FitnessGoal::minimize(1.0).is_ok());
/// ```
#[derive(Debug, thiserror::Error)]
#[error("fitness goal threshold must be between 0.0 and 1.0, got {0}")]
pub struct ThresholdOutOfRange(f64);

impl FitnessGoal {
    /// Creates a minimization goal with the given threshold.
    ///
    /// The optimization will terminate when any individual achieves a fitness value
    /// at or below the threshold. Use this for problems where lower fitness scores
    /// indicate better solutions.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Target fitness value between 0.0 and 1.0. The algorithm stops
    ///   when fitness reaches this value or lower.
    ///
    /// # Returns
    ///
    /// Returns `Ok(FitnessGoal::Minimize)` on success, or `Err(ThresholdOutOfRange)`
    /// if the threshold is outside the valid range [0.0, 1.0].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fx_durable_ga::models::FitnessGoal;
    ///
    /// // Stop when error rate drops to 2%
    /// let goal = FitnessGoal::minimize(0.02)?;
    ///
    /// // Very strict optimization - stop at 0.1% error
    /// let strict_goal = FitnessGoal::minimize(0.001)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn minimize(threshold: f64) -> Result<Self, ThresholdOutOfRange> {
        let threshold = Self::validate(threshold)?;

        Ok(Self::Minimize { threshold })
    }

    /// Creates a maximization goal with the given threshold.
    ///
    /// The optimization will terminate when any individual achieves a fitness value
    /// at or above the threshold. Use this for problems where higher fitness scores
    /// indicate better solutions.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Target fitness value between 0.0 and 1.0. The algorithm stops
    ///   when fitness reaches this value or higher.
    ///
    /// # Returns
    ///
    /// Returns `Ok(FitnessGoal::Maximize)` on success, or `Err(ThresholdOutOfRange)`
    /// if the threshold is outside the valid range [0.0, 1.0].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fx_durable_ga::models::FitnessGoal;
    ///
    /// // Stop when accuracy reaches 90%
    /// let goal = FitnessGoal::maximize(0.90)?;
    ///
    /// // High-precision goal - stop at 99.5% accuracy
    /// let precise_goal = FitnessGoal::maximize(0.995)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
