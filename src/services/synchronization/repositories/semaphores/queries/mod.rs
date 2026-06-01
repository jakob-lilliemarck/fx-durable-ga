mod notify;
mod poll;
mod store;

pub(self) use super::Error;
pub(self) use super::Semaphore;
pub use poll::Filter as PollFilter;

#[cfg(not(test))]
pub(crate) use notify::notify;
#[cfg(test)]
pub(crate) use notify::notify;

#[cfg(not(test))]
pub(crate) use store::store;
#[cfg(test)]
pub(crate) use store::store;

#[cfg(not(any(test, feature = "test-tools")))]
pub(super) use poll::poll;
#[cfg(any(test, feature = "test-tools"))]
pub use poll::poll;
