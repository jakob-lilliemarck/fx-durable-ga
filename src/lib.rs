pub mod migrations;
pub mod models;

#[cfg(not(feature = "migration"))]
mod bootstrap;

#[cfg(not(feature = "migration"))]
mod repositories;

#[cfg(not(feature = "migration"))]
mod services;

#[cfg(not(feature = "migration"))]
pub use bootstrap::bootstrap;

#[cfg(not(feature = "migration"))]
pub use repositories::chainable;

#[cfg(not(feature = "migration"))]
pub use repositories::genotypes::SearchFilter as GenotypesFilter;

#[cfg(not(feature = "migration"))]
pub use services::optimization;

#[cfg(not(feature = "migration"))]
pub use services::optimization::{register_event_handlers, register_job_handlers};

#[cfg(not(feature = "migration"))]
pub use services::genotype_explorer;
