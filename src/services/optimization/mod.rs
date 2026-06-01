mod breeder;
mod errors;
mod events;
mod jobs;
mod optimizer;
pub(super) mod repositories;
pub(crate) use repositories::requests::Read;
#[cfg(any(test, feature = "test-tools"))]
#[allow(unused_imports)]
pub(crate) use repositories::requests::store_request;

mod registrations;
mod service;

pub(crate) use breeder::Breeder;
pub use errors::Error;
pub(crate) use optimizer::OptimizerErased;
pub use optimizer::OptimizerRegistry;
pub use optimizer::{OptimizationService, Optimizer};
pub use registrations::register;
pub use repositories::requests::{
    FitnessGoal, Request, Schedule, SearchRequestsFilter, SelectionError, Selector,
};
pub use service::Service;
