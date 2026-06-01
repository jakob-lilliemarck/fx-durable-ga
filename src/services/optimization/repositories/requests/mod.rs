mod errors;
mod models;
mod queries;
mod registrations;
mod repository;

pub use errors::Error;
pub use registrations::{provide_requests_repository_ro, provide_requests_repository_wr};

pub use models::{FitnessGoal, Request, Schedule, SelectionError, Selector};
pub use queries::SearchRequestsFilter;

#[cfg(any(test, feature = "test-tools"))]
#[allow(unused_imports)]
pub use queries::get_request;

pub(crate) use repository::Read;
pub(crate) use repository::Write;
pub(crate) use repository::WriteTx;

#[cfg(any(test, feature = "test-tools"))]
#[allow(unused_imports)]
pub(crate) use queries::store_request;
