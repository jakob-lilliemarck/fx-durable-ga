mod errors;
mod service;

pub mod encoder;
pub use errors::Error;
pub use service::{EncodeInput, ModelConfig, Service, TrainModelConfig};
