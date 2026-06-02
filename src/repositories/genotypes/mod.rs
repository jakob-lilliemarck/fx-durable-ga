mod errors;
mod models;
mod queries;
mod repository;

pub mod registrations;

pub use errors::Error;
pub use models::Error as GenotypeError;
pub use models::{Genotype, GenotypePopulation, Identifiable, TypeName};
pub use registrations::register;

pub(crate) use models::Population;

pub(crate) use repository::Read;
pub(crate) use repository::Write;
pub(crate) use repository::WriteTx;

#[cfg(any(test, feature = "test-tools"))]
#[allow(unused_imports)]
pub use queries::{get_ancestors, get_descendants, get_population, search_genotypes};

#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use queries::store_genotypes;

pub use queries::SearchGenotypesFilter;
