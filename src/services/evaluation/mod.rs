mod errors;
mod events;
mod registrations;
pub mod repositories;
mod service;

pub use errors::Error;
pub use events::GenotypeEvaluatedEvent;
pub use registrations::register;
pub use repositories::evaluations::AggregatedFitness;
pub use repositories::evaluations::Evaluation;
pub use repositories::evaluations::EvaluationPopulation;
pub use repositories::evaluations::GetAggregatedFitnessFilter;
pub use repositories::evaluations::GetEvaluationStatsFilter;
pub use repositories::evaluations::SearchEvaluationsFilter;
pub use service::Evaluator;
pub(crate) use service::SHUTDOWN_SEMAPHORE;
pub use service::Service;
