use crate::infrastructure::db;
use crate::infrastructure::di::{Container, ProviderResult};
use futures::future::BoxFuture;
use std::sync::Arc;

pub fn provide_semaphores_repository(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Repository>>> {
    Box::pin(async {
        let wr = c.get::<db::WritePool>().await?;
        Ok(Arc::new(super::Repository::new(wr.pool)))
    })
}
