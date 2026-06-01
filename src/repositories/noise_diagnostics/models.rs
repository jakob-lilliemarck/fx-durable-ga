use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("probe_population_size must be greater than 0, was {0}")]
    PopulationSizeNotGreaterThanZero(i32),
    #[error("probe_min_evaluations must be greater than 0, was {0}")]
    MinNotGreaterThanZero(i32),
    #[error("probe_max_evaluations must be greater than 0, , was {0}")]
    MaxNotGreaterThanZero(i32),
    #[error(
        "probe_max_evaluations must be equal or greater than probe_min_evaluations, got min: {min}, max: {max}"
    )]
    MaxLessThanMin { min: i32, max: i32 },
}

#[derive(Debug)]
pub struct NoiseDiagnosticConfig {
    pub(super) id: Uuid,
    pub(super) optimization_type_name: String,
    pub(super) probe_population_size: i32,
    pub(super) probe_min_evaluations: i32,
    pub(super) probe_max_evaluations: i32,
    pub(super) budget_id: Uuid,
}

/// Relates evaluations with a specific diagnostic run
pub struct NoiseDiagnosticEvaluation {
    noise_diagnostic_run_id: Uuid,
    evaluation_id: Uuid,
}

#[derive(Debug)]
pub struct NoiseEstimate {
    optimization_type_name: String,
    diagnostic_run_id: Uuid,
    probe_count: i64,
    evaluation_count: i64,
    std_dev: f64,
}

impl NoiseDiagnosticConfig {
    pub fn new(
        optimization_type_name: &str,
        probe_population_size: i32,
        probe_min_evaluations: i32,
        probe_max_evaluations: i32,
        budget_id: &Uuid,
    ) -> Result<Self, ValidationError> {
        let id = Uuid::now_v7();
        Self::validate_probe_population_size(probe_population_size)?;
        Self::validate_probe_min_evaluations(probe_min_evaluations)?;
        Self::validate_probe_max_evaluations(probe_max_evaluations, probe_min_evaluations)?;

        Ok(Self {
            id,
            optimization_type_name: optimization_type_name.to_string(),
            probe_population_size,
            probe_min_evaluations,
            probe_max_evaluations,
            budget_id: *budget_id,
        })
    }

    fn validate_probe_population_size(probe_population_size: i32) -> Result<i32, ValidationError> {
        if probe_population_size <= 0 {
            return Err(ValidationError::PopulationSizeNotGreaterThanZero(
                probe_population_size,
            ));
        }

        Ok(probe_population_size)
    }

    fn validate_probe_min_evaluations(probe_min_evaluations: i32) -> Result<i32, ValidationError> {
        if probe_min_evaluations <= 0 {
            return Err(ValidationError::MinNotGreaterThanZero(
                probe_min_evaluations,
            ));
        }

        Ok(probe_min_evaluations)
    }

    fn validate_probe_max_evaluations(
        probe_max_evaluations: i32,
        probe_min_evaluations: i32,
    ) -> Result<i32, ValidationError> {
        if probe_max_evaluations <= 0 {
            return Err(ValidationError::MaxNotGreaterThanZero(
                probe_max_evaluations,
            ));
        }

        if probe_max_evaluations < probe_min_evaluations {
            return Err(ValidationError::MaxLessThanMin {
                min: probe_min_evaluations,
                max: probe_max_evaluations,
            });
        }

        Ok(probe_max_evaluations)
    }
}
