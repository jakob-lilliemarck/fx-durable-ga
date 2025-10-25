use crate::models::{Genotype, Morphology};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Applies linear decay from upper to lower bounds based on progress.
fn decay_linear(upper: f64, lower: f64, progress: f64, multiplier: f64) -> f64 {
    lower + (upper - lower) * (1.0 - progress * multiplier).max(0.0)
}

/// Applies exponential decay from upper to lower bounds based on progress.
fn decay_exponential(upper: f64, lower: f64, progress: f64, multiplier: f64, exponent: i32) -> f64 {
    lower + (upper - lower) * (1.0 - progress * multiplier).max(0.0).powi(exponent)
}

// ============================================================
// Decay
// ============================================================

/// Strategy for decaying parameter values over optimization progress.
///
/// Decay strategies control how parameters like temperature and mutation rate change
/// as the genetic algorithm progresses. This allows the algorithm to start with
/// exploratory behavior (high values) and gradually shift to exploitative behavior
/// (low values) as it converges on optimal solutions.
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::{Temperature, MutationRate};
///
/// // Start aggressive, end conservative
/// let temp = Temperature::linear(0.8, 0.1, 1.0)?;
///
/// // Exponential cooling for faster convergence
/// let rate = MutationRate::exponential(0.5, 0.05, 1.0, 3)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decay {
    /// Parameter remains unchanged throughout optimization.
    ///
    /// Use when you want consistent behavior or when you're unsure
    /// about the optimal decay strategy for your problem.
    Constant,

    /// Parameter decreases linearly from upper to lower bound.
    ///
    /// Formula: `lower + (upper - lower) * (1.0 - progress * multiplier)`
    ///
    /// - `lower`: Final value when fully converged (progress = 1.0)
    /// - `multiplier`: Controls decay speed (1.0 = normal, >1.0 = faster, <1.0 = slower)
    Linear { lower: f64, multiplier: f64 },

    /// Parameter decreases exponentially, providing rapid early decay that slows over time.
    ///
    /// Formula: `lower + (upper - lower) * (1.0 - progress * multiplier)^exponent`
    ///
    /// - `lower`: Final value when fully converged
    /// - `multiplier`: Controls decay timing (higher = faster initial decay)
    /// - `exponent`: Controls curve steepness (higher = more aggressive early decay)
    Exponential {
        lower: f64,
        multiplier: f64,
        exponent: i32,
    },
}

impl Decay {
    /// Applies the decay strategy to get the current parameter value.
    fn apply(&self, upper: f64, progress: f64) -> f64 {
        match self {
            Decay::Constant => upper,
            Decay::Linear { lower, multiplier } => {
                decay_linear(upper, *lower, progress, *multiplier)
            }
            Decay::Exponential {
                lower,
                multiplier,
                exponent,
            } => decay_exponential(upper, *lower, progress, *multiplier, *exponent),
        }
    }
}

// ============================================================
// Temperature
// ============================================================

/// Controls mutation step size in the genetic algorithm.
///
/// Temperature determines how large the mutations can be when modifying genes.
/// Higher temperatures allow larger jumps in the search space (exploration),
/// while lower temperatures make smaller, more refined changes (exploitation).
///
/// # When to Use Different Values
///
/// - **High (0.7-1.0)**: Early exploration, large search spaces, getting unstuck from local optima
/// - **Medium (0.3-0.7)**: Balanced exploration/exploitation, most general-purpose scenarios  
/// - **Low (0.0-0.3)**: Fine-tuning, convergence phase, small search spaces
///
/// # Decay Strategies
///
/// - **Constant**: Consistent behavior throughout optimization
/// - **Linear**: Gradual transition from exploration to exploitation
/// - **Exponential**: Rapid early exploration followed by fine-tuning
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::Temperature;
///
/// // Simple constant temperature
/// let temp = Temperature::constant(0.5)?;
///
/// // Start exploratory, end conservative
/// let temp = Temperature::linear(0.9, 0.1, 1.0)?;
///
/// // Aggressive early exploration with rapid cooling
/// let temp = Temperature::exponential(0.8, 0.05, 1.2, 3)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Temperature {
    value: f64,
    decay: Decay,
}

#[derive(Debug, thiserror::Error)]
pub enum TemperatureError {
    #[error("temperature value must be between 0.0 and 1.0, got: {0}")]
    ValueOutOfRange(f64),
    #[error("temperature lower bound ({lower}) must be <= upper bound ({upper})")]
    InvalidBounds { lower: f64, upper: f64 },
}

impl Temperature {
    /// Creates a constant temperature that remains unchanged throughout optimization.
    ///
    /// # Parameters
    /// - `value`: Temperature value in [0.0, 1.0]. Higher values enable larger mutation steps.
    ///
    /// # When to Use
    /// - Simple optimization problems
    /// - When you want predictable, consistent mutation behavior
    /// - Testing and debugging genetic algorithms
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::Temperature;
    ///
    /// let low_temp = Temperature::constant(0.2)?;    // Conservative mutations
    /// let high_temp = Temperature::constant(0.8)?;   // Aggressive mutations
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn constant(value: f64) -> Result<Self, TemperatureError> {
        let value = Self::validate(value)?;

        Ok(Self {
            value,
            decay: Decay::Constant,
        })
    }

    /// Validates that temperature value is within [0.0, 1.0].
    fn validate(value: f64) -> Result<f64, TemperatureError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(TemperatureError::ValueOutOfRange(value));
        }

        Ok(value)
    }

    /// Creates a temperature that decays linearly from high to low values.
    ///
    /// Linear decay provides a smooth, predictable transition from exploration
    /// to exploitation as the algorithm progresses.
    ///
    /// # Parameters
    /// - `upper`: Starting temperature (0.0-1.0) for early exploration
    /// - `lower`: Ending temperature (0.0-1.0) for final convergence  
    /// - `multiplier`: Decay speed (1.0 = normal, >1.0 = faster, <1.0 = slower)
    ///
    /// # When to Use
    /// - Most general-purpose optimization problems
    /// - When you want predictable exploration-to-exploitation transition
    /// - Medium to large search spaces where gradual refinement helps
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::Temperature;
    ///
    /// // Standard linear cooling
    /// let temp = Temperature::linear(0.8, 0.1, 1.0)?;
    ///
    /// // Slower cooling for complex problems
    /// let temp = Temperature::linear(0.7, 0.2, 0.5)?;
    ///
    /// // Faster cooling for quick convergence
    /// let temp = Temperature::linear(0.9, 0.05, 1.5)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn linear(upper: f64, lower: f64, multiplier: f64) -> Result<Self, TemperatureError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;

        if lower > upper {
            return Err(TemperatureError::InvalidBounds { lower, upper });
        }

        Ok(Self {
            value: upper,
            decay: Decay::Linear { lower, multiplier },
        })
    }

    /// Creates a temperature that decays exponentially for rapid early cooling.
    ///
    /// Exponential decay provides aggressive early exploration followed by
    /// fine-tuned exploitation. The algorithm quickly transitions from broad
    /// search to focused refinement.
    ///
    /// # Parameters
    /// - `upper`: Starting temperature (0.0-1.0) for initial exploration
    /// - `lower`: Ending temperature (0.0-1.0) for final fine-tuning
    /// - `multiplier`: Controls when decay begins (higher = earlier/faster)
    /// - `exponent`: Controls decay curve steepness (higher = more aggressive)
    ///
    /// # When to Use
    /// - Complex landscapes with many local optima
    /// - When you need rapid convergence after initial exploration
    /// - Problems where early diversity is crucial
    /// - Time-constrained optimization
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::Temperature;
    ///
    /// // Moderate exponential cooling
    /// let temp = Temperature::exponential(0.8, 0.1, 1.0, 2)?;
    ///
    /// // Aggressive early exploration
    /// let temp = Temperature::exponential(0.9, 0.05, 1.2, 4)?;
    ///
    /// // Gentle exponential curve
    /// let temp = Temperature::exponential(0.6, 0.2, 0.8, 2)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn exponential(
        upper: f64,
        lower: f64,
        multiplier: f64,
        exponent: i32,
    ) -> Result<Self, TemperatureError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;

        if lower > upper {
            return Err(TemperatureError::InvalidBounds { lower, upper });
        }

        Ok(Self {
            value: upper,
            decay: Decay::Exponential {
                lower,
                multiplier,
                exponent,
            },
        })
    }

    /// Gets the current temperature value based on optimization progress.
    fn get(&self, progress: f64) -> f64 {
        self.decay.apply(self.value, progress)
    }
}

// ============================================================
// MutationRate
// ============================================================

/// Controls the probability that each gene will be mutated during reproduction.
///
/// Mutation rate determines how frequently genes are changed when creating offspring.
/// Higher rates provide more genetic diversity but may disrupt good solutions,
/// while lower rates preserve good solutions but may limit exploration.
///
/// # When to Use Different Values
///
/// - **High (0.7-1.0)**: Dense problems, escaping local optima, early exploration
/// - **Medium (0.3-0.7)**: Balanced search, most general-purpose applications
/// - **Low (0.0-0.3)**: Preserving good solutions, fine-tuning, convergence phase
///
/// # Interaction with Population Size
///
/// - **Small populations**: Use higher mutation rates to maintain diversity
/// - **Large populations**: Can use lower mutation rates as diversity emerges naturally
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::MutationRate;
///
/// // Conservative mutation for stable problems
/// let rate = MutationRate::constant(0.1)?;
///
/// // Start diverse, converge to precision
/// let rate = MutationRate::linear(0.6, 0.05, 1.0)?;
///
/// // Rapid diversity reduction
/// let rate = MutationRate::exponential(0.8, 0.02, 1.5, 3)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRate {
    value: f64,
    decay: Decay,
}

#[derive(Debug, thiserror::Error)]
pub enum MutationRateError {
    #[error("mutation rate value must be between 0.0 and 1.0, got: {0}")]
    ValueOutOfRange(f64),
    #[error("mutation rate lower bound ({lower}) must be <= upper bound ({upper})")]
    InvalidBounds { lower: f64, upper: f64 },
}

impl MutationRate {
    /// Creates a constant mutation rate that remains unchanged throughout optimization.
    ///
    /// # Parameters
    /// - `value`: Mutation probability per gene in [0.0, 1.0]. Higher values increase genetic diversity.
    ///
    /// # When to Use
    /// - Simple problems with stable optima
    /// - Baseline testing and comparison
    /// - When optimal mutation rate is already known
    /// - Problems requiring consistent exploration level
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::MutationRate;
    ///
    /// let conservative = MutationRate::constant(0.05)?;  // 5% mutation chance
    /// let balanced = MutationRate::constant(0.3)?;       // 30% mutation chance
    /// let aggressive = MutationRate::constant(0.7)?;     // 70% mutation chance
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn constant(value: f64) -> Result<Self, MutationRateError> {
        let value = Self::validate(value)?;

        Ok(Self {
            value,
            decay: Decay::Constant,
        })
    }

    /// Validates that mutation rate value is within [0.0, 1.0].
    fn validate(value: f64) -> Result<f64, MutationRateError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(MutationRateError::ValueOutOfRange(value));
        }

        Ok(value)
    }

    /// Creates a mutation rate that decays linearly to reduce disruption over time.
    ///
    /// Linear decay starts with high genetic diversity and gradually reduces
    /// mutations to preserve promising solutions as they emerge.
    ///
    /// # Parameters
    /// - `upper`: Starting mutation rate (0.0-1.0) for early diversity
    /// - `lower`: Ending mutation rate (0.0-1.0) for solution preservation
    /// - `multiplier`: Decay speed (1.0 = normal, >1.0 = faster, <1.0 = slower)
    ///
    /// # When to Use
    /// - Most optimization problems benefit from this approach
    /// - When good solutions emerge gradually over time
    /// - Problems where early diversity and later stability both matter
    /// - Balanced exploration-exploitation scenarios
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::MutationRate;
    ///
    /// // Standard diversity-to-stability transition
    /// let rate = MutationRate::linear(0.5, 0.1, 1.0)?;
    ///
    /// // Slower transition for complex problems
    /// let rate = MutationRate::linear(0.6, 0.2, 0.7)?;
    ///
    /// // Rapid stabilization
    /// let rate = MutationRate::linear(0.8, 0.05, 1.4)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn linear(upper: f64, lower: f64, multiplier: f64) -> Result<Self, MutationRateError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;

        if lower > upper {
            return Err(MutationRateError::InvalidBounds { lower, upper });
        }

        Ok(Self {
            value: upper,
            decay: Decay::Linear { lower, multiplier },
        })
    }

    /// Creates a mutation rate that decays exponentially for rapid convergence.
    ///
    /// Exponential decay maintains high genetic diversity initially, then rapidly
    /// reduces mutations to lock in promising solutions. This provides an intense
    /// early search followed by strong solution preservation.
    ///
    /// # Parameters
    /// - `upper`: Starting mutation rate (0.0-1.0) for initial diversity
    /// - `lower`: Ending mutation rate (0.0-1.0) for final stability
    /// - `multiplier`: Controls timing of decay (higher = earlier/faster reduction)
    /// - `exponent`: Controls decay aggressiveness (higher = more dramatic curve)
    ///
    /// # When to Use
    /// - Time-constrained optimization requiring fast convergence
    /// - Problems with clear early vs. late phases
    /// - When disruption of good solutions is particularly costly
    /// - Search spaces where early exploration is critical
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::MutationRate;
    ///
    /// // Rapid diversity reduction
    /// let rate = MutationRate::exponential(0.7, 0.05, 1.0, 3)?;
    ///
    /// // Extreme early diversity, quick stabilization
    /// let rate = MutationRate::exponential(0.9, 0.01, 1.3, 4)?;
    ///
    /// // Gentler exponential curve
    /// let rate = MutationRate::exponential(0.5, 0.1, 0.8, 2)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn exponential(
        upper: f64,
        lower: f64,
        multiplier: f64,
        exponent: i32,
    ) -> Result<Self, MutationRateError> {
        let upper = Self::validate(upper)?;
        let lower = Self::validate(lower)?;

        if lower > upper {
            return Err(MutationRateError::InvalidBounds { lower, upper });
        }

        Ok(Self {
            value: upper,
            decay: Decay::Exponential {
                lower,
                multiplier,
                exponent,
            },
        })
    }

    /// Gets the current mutation rate value based on optimization progress.
    fn get(&self, progress: f64) -> f64 {
        self.decay.apply(self.value, progress)
    }
}

// ============================================================
// Mutagen
// ============================================================

/// Orchestrates genetic mutation by combining temperature and mutation rate strategies.
///
/// The `Mutagen` is the core component that controls how genetic algorithms modify
/// candidate solutions. It combines two key parameters:
///
/// - **Temperature**: Controls mutation step size (how big changes can be)
/// - **Mutation Rate**: Controls mutation frequency (how often changes occur)
///
/// Together, these parameters determine the exploration/exploitation balance
/// throughout the optimization process.
///
/// # Key Concepts
///
/// ## Exploration vs. Exploitation
/// - **High temperature + High mutation rate**: Aggressive exploration
/// - **Low temperature + Low mutation rate**: Focused exploitation  
/// - **Mixed strategies**: Balanced search behavior
///
/// ## Common Patterns
/// - **Simulated Annealing**: Start hot and diverse, end cool and stable
/// - **Constant Search**: Maintain consistent behavior throughout
/// - **Rapid Convergence**: Quickly transition from exploration to exploitation
///
/// # Configuration Examples
///
/// ```rust
/// use fx_durable_ga::models::{Mutagen, Temperature, MutationRate};
///
/// // Simple constant behavior - good for testing
/// let simple = Mutagen::constant(0.3, 0.1)?;
///
/// // Classic simulated annealing approach
/// let annealing = Mutagen::new(
///     Temperature::exponential(0.8, 0.1, 1.0, 2)?,
///     MutationRate::linear(0.5, 0.05, 1.0)?
/// );
///
/// // Balanced linear decay for most problems
/// let balanced = Mutagen::new(
///     Temperature::linear(0.6, 0.2, 1.0)?,
///     MutationRate::linear(0.4, 0.1, 1.0)?
/// );
///
/// // Rapid early exploration, then fine-tuning
/// let rapid = Mutagen::new(
///     Temperature::exponential(0.9, 0.05, 1.2, 3)?,
///     MutationRate::exponential(0.7, 0.02, 1.1, 3)?
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Choosing the Right Strategy
///
/// | Problem Type | Temperature | Mutation Rate | Rationale |
/// |--------------|-------------|---------------|----------|
/// | Simple/Known | Constant | Constant | Predictable behavior |
/// | General Purpose | Linear decay | Linear decay | Balanced approach |
/// | Complex/Time-constrained | Exponential | Exponential | Rapid convergence |
/// | Large search space | High initial | High initial | Need exploration |
/// | Fine-tuning | Low constant | Low constant | Preserve solutions |
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutagen {
    mutation_rate: MutationRate,
    temperature: Temperature,
}

/// Errors that can occur when configuring genetic algorithm mutation parameters.
///
/// These errors provide specific feedback about invalid configuration values,
/// helping you quickly identify and fix parameter setup issues.
///
/// # Common Scenarios
///
/// - **Out of Range**: Parameter values must be within [0.0, 1.0]
/// - **Invalid Bounds**: Lower bounds must be ≤ upper bounds for decay strategies
/// - **Configuration Conflicts**: Incompatible parameter combinations
///
/// # Examples
///
/// ```rust
/// use fx_durable_ga::models::{Mutagen, Temperature, MutationRate};
///
/// // This will return MutagenError::Temperature
/// let result = Mutagen::constant(1.5, 0.3); // temperature > 1.0
/// assert!(result.is_err());
///
/// // This will return MutagenError::MutationRate
/// let result = Mutagen::constant(0.5, -0.1); // negative mutation rate
/// assert!(result.is_err());
///
/// // Invalid bounds in decay strategies
/// let result = Temperature::linear(0.2, 0.8, 1.0); // lower > upper
/// assert!(result.is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Error Handling
///
/// ```rust
/// use fx_durable_ga::models::{Mutagen, MutagenError};
///
/// match Mutagen::constant(1.5, 0.3) {
///     Ok(mutagen) => {
///         // Use the configured mutagen
///     }
///     Err(MutagenError::Temperature(e)) => {
///         eprintln!("Temperature configuration error: {e}");
///         // Handle temperature-specific error
///     }
///     Err(MutagenError::MutationRate(e)) => {
///         eprintln!("Mutation rate configuration error: {e}");
///         // Handle mutation rate-specific error  
///     }
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, thiserror::Error)]
pub enum MutagenError {
    /// Mutation rate parameter configuration is invalid.
    ///
    /// This occurs when mutation rate values are outside [0.0, 1.0] or when
    /// decay strategy bounds are inconsistent (lower > upper).
    #[error("Mutation rate error: {0}")]
    MutationRate(#[from] MutationRateError),

    /// Temperature parameter configuration is invalid.
    ///
    /// This occurs when temperature values are outside [0.0, 1.0] or when
    /// decay strategy bounds are inconsistent (lower > upper).
    #[error("Temperature error: {0}")]
    Temperature(#[from] TemperatureError),
}

impl Mutagen {
    /// Creates a new mutagen by combining temperature and mutation rate strategies.
    ///
    /// This is the primary constructor for creating sophisticated mutation behaviors
    /// by combining different decay strategies for temperature and mutation rate.
    ///
    /// # Parameters
    /// - `temperature`: Controls mutation step size over time
    /// - `mutation_rate`: Controls mutation frequency over time
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::{Mutagen, Temperature, MutationRate};
    ///
    /// // Mixed strategies: exponential temperature, linear mutation rate
    /// let mutagen = Mutagen::new(
    ///     Temperature::exponential(0.8, 0.1, 1.0, 2)?,
    ///     MutationRate::linear(0.4, 0.05, 1.0)?
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(temperature: Temperature, mutation_rate: MutationRate) -> Self {
        Self {
            temperature,
            mutation_rate,
        }
    }

    /// Creates a mutagen with constant temperature and mutation rate values.
    ///
    /// This convenience method creates the simplest possible mutagen configuration
    /// with unchanging parameters. Use this for baseline testing, simple problems,
    /// or when you want predictable mutation behavior.
    ///
    /// # Parameters
    /// - `temperature_value`: Fixed temperature in [0.0, 1.0] for mutation step size
    /// - `mutation_rate_value`: Fixed mutation rate in [0.0, 1.0] for gene change frequency
    ///
    /// # When to Use
    /// - Testing and debugging genetic algorithms
    /// - Simple optimization problems with known characteristics  
    /// - Baseline comparisons against adaptive strategies
    /// - When optimal parameters are already determined
    ///
    /// # Examples
    /// ```rust
    /// use fx_durable_ga::models::Mutagen;
    ///
    /// let conservative = Mutagen::constant(0.2, 0.05)?;  // Small, rare mutations
    /// let balanced = Mutagen::constant(0.5, 0.3)?;       // Medium mutations
    /// let aggressive = Mutagen::constant(0.8, 0.7)?;     // Large, frequent mutations
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn constant(
        temperature_value: f64,
        mutation_rate_value: f64,
    ) -> Result<Self, MutagenError> {
        Ok(Self::new(
            Temperature::constant(temperature_value)?,
            MutationRate::constant(mutation_rate_value)?,
        ))
    }

    /// Mutates a genotype in place based on current temperature and mutation rate.
    #[instrument(level = "debug", skip(self, rng, genotype, morphology), fields(progress = progress, genotype_id = %genotype.id))]
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
            // Apply mutation based on current mutation rate
            if rng.random_range(0.0..1.0) < mutation_rate {
                // Temperature controls mutation step size: higher = larger jumps
                let max_step = (1.0 + (bounds.steps() as f64 * temperature)) as i64;

                // Choose random direction and step size
                let direction = if rng.random_bool(0.5) { 1 } else { -1 };
                let step = rng.random_range(1..=max_step);

                // Apply mutation and clamp to valid range
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
    fn it_validates_temperature_and_mutation_rate_bounds() {
        // Temperature validation
        assert!(Temperature::constant(-0.1).is_err());
        assert!(Temperature::constant(1.5).is_err());
        assert!(Temperature::linear(-0.1, 0.0, 1.0).is_err());
        assert!(Temperature::exponential(1.5, 0.0, 1.0, 2).is_err());

        // MutationRate validation
        assert!(MutationRate::constant(-0.1).is_err());
        assert!(MutationRate::constant(1.5).is_err());
        assert!(MutationRate::linear(-0.1, 0.0, 1.0).is_err());
        assert!(MutationRate::exponential(1.5, 0.0, 1.0, 2).is_err());

        // Lower > upper validation for MutationRate
        assert!(MutationRate::linear(0.3, 0.5, 1.0).is_err());
        assert!(MutationRate::exponential(0.2, 0.4, 1.0, 2).is_err());

        // Lower > upper validation for Temperature
        assert!(Temperature::linear(0.3, 0.5, 1.0).is_err());
        assert!(Temperature::exponential(0.2, 0.4, 1.0, 2).is_err());
    }

    #[test]
    fn it_applies_constant_decay() {
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
    fn it_applies_linear_decay() {
        let temp = Temperature::linear(1.0, 0.0, 1.0).unwrap();
        let rate = MutationRate::linear(1.0, 0.0, 1.0).unwrap();

        // Linear decay: value * (1.0 - progress * multiplier)
        assert_eq!(temp.get(0.0), 1.0); // No progress
        assert_eq!(temp.get(0.5), 0.5); // Half progress
        assert_eq!(temp.get(1.0), 0.0); // Full progress
        assert_eq!(temp.get(1.5), 0.0); // Over-progress (clamped)

        assert_eq!(rate.get(0.0), 1.0);
        assert_eq!(rate.get(0.5), 0.5);
        assert_eq!(rate.get(1.0), 0.0);
        assert_eq!(rate.get(1.5), 0.0);
    }

    #[test]
    fn it_applies_exponential_decay() {
        let temp = Temperature::exponential(1.0, 0.0, 1.0, 2).unwrap();
        let rate = MutationRate::exponential(1.0, 0.0, 1.0, 2).unwrap();

        // Exponential decay: value * (1.0 - progress * multiplier)^exponent
        assert_eq!(temp.get(0.0), 1.0); // No progress
        assert_eq!(temp.get(0.5), 0.25); // Quadratic: (1.0 - 0.5)^2 = 0.25
        assert_eq!(temp.get(1.0), 0.0); // Full progress

        assert_eq!(rate.get(0.0), 1.0);
        assert_eq!(rate.get(0.5), 0.25);
        assert_eq!(rate.get(1.0), 0.0);
    }

    #[test]
    fn it_mutates() {
        let mut rng = StdRng::seed_from_u64(42);
        let morphology = get_test_morphology();
        let mut genotype = get_test_genotype();

        // Create test data
        let mutagen = Mutagen::new(
            Temperature::constant(0.1).expect("temperature is in range"), // Low temp = small steps
            MutationRate::constant(1.0).expect("mutation_rate is in range"), // 100% mutation rate
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
            Temperature::constant(1.0).expect("temperature is in range"),
            MutationRate::constant(0.0).expect("mutation_rate is in range"), // 0% mutation rate
        );

        let original_genome = genotype.genome.clone();

        mutagen.mutate(&mut rng, &mut genotype, &morphology, 0.0);

        // Should be unchanged with 0% mutation rate
        assert_eq!(genotype.genome, original_genome);
    }

    #[test]
    fn it_composes_different_decay_strategies() {
        // Test that we can mix different decay types
        let mutagen = Mutagen::new(
            Temperature::linear(0.8, 0.1, 0.9).unwrap(), // Linear temperature decay
            MutationRate::exponential(0.5, 0.0, 1.0, 2).unwrap(), // Exponential mutation rate decay
        );

        // Verify the composition works correctly
        assert_eq!(mutagen.temperature.get(0.0), 0.8);
        assert_eq!(mutagen.mutation_rate.get(0.0), 0.5);
        // Exponential decay: lower + (upper - lower) * (1.0 - progress * multiplier)^exponent
        // 0.0 + (0.5 - 0.0) * (1.0 - 0.5 * 1.0)^2 = 0.5 * (0.5)^2 = 0.5 * 0.25 = 0.125
        assert_eq!(mutagen.mutation_rate.get(0.5), 0.125);

        // Linear temperature decay: lower + (upper - lower) * (1.0 - progress * multiplier)
        // 0.1 + (0.8 - 0.1) * (1.0 - 0.5 * 0.9) = 0.1 + 0.7 * 0.55 = 0.1 + 0.385 = 0.485
        assert!((mutagen.temperature.get(0.5) - 0.485).abs() < 1e-10);
    }

    #[test]
    fn it_validates_mutagen_constant_parameters() {
        // Test temperature validation error
        assert!(Mutagen::constant(-0.1, 0.5).is_err());
        assert!(Mutagen::constant(1.5, 0.5).is_err());

        // Test mutation rate validation error
        assert!(Mutagen::constant(0.5, -0.1).is_err());
        assert!(Mutagen::constant(0.5, 1.5).is_err());

        // Test successful case
        assert!(Mutagen::constant(0.5, 0.3).is_ok());
    }
}
