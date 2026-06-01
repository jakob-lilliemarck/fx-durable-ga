mod check_if_generation_exists;
mod get_ancestors;
mod get_descendants;
mod get_population;
mod search_genotypes;
mod store_genotypes;

#[cfg(test)]
pub mod test_tools;

pub use get_ancestors::get_ancestors;
pub use get_descendants::get_descendants;
pub use search_genotypes::{SearchGenotypesFilter, search_genotypes};

pub(crate) use store_genotypes::store_genotypes;

// FIXME:
// Generation-related data should eventually move out of genotypes,
// not all genotypes will have a generation (diagnostics etc.)
pub(crate) use check_if_generation_exists::check_if_generation_exists;
pub use get_population::get_population;
