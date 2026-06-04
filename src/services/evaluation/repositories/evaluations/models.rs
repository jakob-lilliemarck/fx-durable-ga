use chrono::{DateTime, Utc};
use sqlx::FromRow;
use tracing::instrument;
use uuid::Uuid;

/// Evaluation-level population stats for a request (no genotype data).
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct EvaluationPopulation {
    pub(crate) evaluated_genotypes: i64,
    pub(crate) min_fitness: Option<f64>,
    pub(crate) max_fitness: Option<f64>,
}

impl EvaluationPopulation {
    /// Creates an empty stats object (zero evaluated, no min/max).
    pub fn empty() -> Self {
        Self {
            evaluated_genotypes: 0,
            min_fitness: None,
            max_fitness: None,
        }
    }

    pub fn evaluated_genotypes(&self) -> i64 {
        self.evaluated_genotypes
    }

    pub fn min_fitness(&self) -> Option<f64> {
        self.min_fitness
    }

    pub fn max_fitness(&self) -> Option<f64> {
        self.max_fitness
    }
}

/// Aggregate timing statistics for a set of evaluations.
pub struct TimingsSummary {
    pub(crate) records: i64,
    pub(crate) min: std::time::Duration,
    pub(crate) max: std::time::Duration,
    pub(crate) avg: std::time::Duration,
    pub(crate) percentiles: Vec<std::time::Duration>,
}

impl TimingsSummary {
    pub fn records(&self) -> i64 {
        self.records
    }

    pub fn min(&self) -> &std::time::Duration {
        &self.min
    }

    pub fn max(&self) -> &std::time::Duration {
        &self.max
    }

    pub fn avg(&self) -> &std::time::Duration {
        &self.avg
    }

    pub fn percentiles(&self) -> &[std::time::Duration] {
        &self.percentiles
    }
}

/// An evaluation result for a single genotype.
#[derive(Debug, Clone, FromRow)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Evaluation {
    pub(crate) id: Uuid,
    pub(crate) genotype_id: Uuid,
    pub(crate) fitness: f64,
    pub(crate) started_at: Option<DateTime<Utc>>,
    pub(crate) completed_at: Option<DateTime<Utc>>,
    pub(crate) evaluated_by: Option<Uuid>,
    pub(crate) group_id: Uuid,
    pub(crate) reason: String,
    pub(crate) generated_at: Option<DateTime<Utc>>,
}

impl Evaluation {
    #[instrument(level = "debug", fields(
        genotype_id = %genotype_id,
        fitness = fitness,
        started_at = ?started_at,
        completed_at = ?completed_at,
        evaluated_by = ?evaluated_by
    ))]
    /// Creates an evaluation for a genotype with the given fitness and timing data.
    pub(crate) fn new(
        genotype_id: Uuid,
        group_id: Uuid,
        reason: String,
        fitness: f64,
        started_at: Option<DateTime<Utc>>,
        completed_at: Option<DateTime<Utc>>,
        evaluated_by: Option<Uuid>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            genotype_id,
            group_id,
            reason,
            fitness,
            started_at,
            completed_at,
            evaluated_by,
            generated_at: None,
        }
    }

    /// Sets the timestamp when the evaluated genotype was generated.
    pub fn with_generated_at(mut self, generated_at: DateTime<Utc>) -> Self {
        self.generated_at = Some(generated_at);
        self
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn genotype_id(&self) -> &Uuid {
        &self.genotype_id
    }

    pub fn fitness(&self) -> f64 {
        self.fitness
    }

    pub fn started_at(&self) -> &Option<DateTime<Utc>> {
        &self.started_at
    }

    pub fn completed_at(&self) -> &Option<DateTime<Utc>> {
        &self.completed_at
    }

    pub fn evaluated_by(&self) -> &Option<Uuid> {
        &self.evaluated_by
    }

    pub fn group_id(&self) -> Uuid {
        self.group_id
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }

    pub fn generated_at(&self) -> &Option<DateTime<Utc>> {
        &self.generated_at
    }
}
