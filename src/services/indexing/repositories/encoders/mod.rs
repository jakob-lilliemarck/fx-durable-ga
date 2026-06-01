mod errors;
mod queries;
mod registrations;
mod repository;

pub use errors::Error;
pub use registrations::provide_encoders_repository_ro;
pub use registrations::provide_encoders_repository_wr;

#[cfg(any(test, feature = "test-tools"))]
#[allow(unused_imports)]
pub use queries::{get_available_encoder_digests, get_encoder};

pub use queries::Digest;
pub use queries::Error as EncoderDigestError;

pub(crate) use queries::Encoder;
pub(crate) use queries::EncoderAvailability;
pub(crate) use repository::{Read, Write, WriteTx};

#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use queries::{store_encoder, store_encoder_availability};
