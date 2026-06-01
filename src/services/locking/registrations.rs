use crate::infrastructure::{
    db,
    di::{Container, ProviderError, ProviderResult},
};
use futures::future::BoxFuture;
use std::sync::Arc;

#[cfg(test)]
use futures::{
    channel::{mpsc, oneshot},
    lock::Mutex,
};

#[cfg(test)]
#[allow(dead_code)]
fn provide_locking_tx_control(
    _: &mut Container,
    tx: Mutex<mpsc::Sender<oneshot::Sender<()>>>,
) -> BoxFuture<'_, ProviderResult<super::service::TxControl>> {
    Box::pin(async { Ok(Arc::new(tx)) })
}

fn provide_locking_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        let wr = c.get::<db::WritePool>().await?;

        let tx = match c.get::<super::service::TxControl>().await {
            Ok(tx) => Ok(Some(tx)),
            Err(ProviderError::NoProvider { .. }) => Ok(None),
            Err(err) => Err(err),
        }?;

        let locking = super::Service::new(wr.pool, tx);

        Ok(Arc::new(locking))
    })
}

/// Registers the locking service with the DI container.
pub fn register(c: &mut Container) {
    c.provide(provide_locking_service);
}
