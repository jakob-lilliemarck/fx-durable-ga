mod errors;
mod events;
mod jobs;
mod service;
mod service_builder;

pub mod encoder;
pub use errors::Error;
pub use events::{EmbeddingCreatedEvent, EncoderTrainedEvent, register_event_handlers};
pub use jobs::register_job_handlers;
pub use service::{ModelConfig, Service, TrainModelConfig};
pub use service_builder::ServiceBuilder;
