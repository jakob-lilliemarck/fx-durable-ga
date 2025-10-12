pub mod gene;
pub mod registry;

pub use gene::{BoundsError, GeneBounds};
pub use registry::{Encodeable, Evaluator, Registry};
