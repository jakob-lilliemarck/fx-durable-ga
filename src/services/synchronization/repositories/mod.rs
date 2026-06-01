pub mod semaphores;

#[cfg(any(test, feature = "test-tools"))]
#[allow(unused_imports)]
pub use semaphores::queries::poll;
