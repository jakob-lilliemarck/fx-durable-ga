pub mod configuration;
pub mod infrastructure;
pub mod migrations;

#[cfg(not(feature = "migration"))]
pub use bootstrap::register;

#[cfg(not(feature = "migration"))]
pub mod views;

#[cfg(not(feature = "migration"))]
pub mod repositories;

#[cfg(not(feature = "migration"))]
pub mod bootstrap;

#[cfg(not(feature = "migration"))]
pub use repositories::genotypes::SearchGenotypesFilter;

#[cfg(not(feature = "migration"))]
pub mod services;

#[cfg(not(feature = "migration"))]
pub mod controllers;

#[cfg(not(feature = "migration"))]
pub mod api_client;

#[cfg(all(not(feature = "migration"), any(test, feature = "test-tools")))]
pub mod test_tools;
