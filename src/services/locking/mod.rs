mod errors;
mod service;

pub mod registrations;

pub use errors::Error;
pub use registrations::register;
pub use service::Service;
