use futures::future::BoxFuture;

/// Provides termination checking capability for long-running evaluations.
/// Allows evaluators to check if they should abort early due to request cancellation.
pub trait Terminated: Send + Sync {
    /// Returns true if the evaluation should be terminated early.
    fn is_terminated(&self) -> BoxFuture<'_, bool>;
}

impl<T: Terminated + ?Sized> Terminated for Box<T> {
    fn is_terminated(&self) -> BoxFuture<'_, bool> {
        (**self).is_terminated()
    }
}
