mod queries;
mod repository;

pub(crate) use queries::{Filter, Order};
pub(crate) use repository::{Error, Gene, Genotype, Repository};
