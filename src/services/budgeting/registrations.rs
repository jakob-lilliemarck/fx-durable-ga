use super::repositories::transactions;
use super::repositories::transactions::registrations::{
    provide_transactions_repository_ro, provide_transactions_repository_wr,
};
use crate::infrastructure::di::{Container, ProviderResult};
use futures::future::BoxFuture;
use std::sync::Arc;

fn provide_budgeting_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        let transactions_wr = c.get::<transactions::Write>().await?;
        let budgeting = Arc::new(super::Service::new(transactions_wr));

        Ok(budgeting)
    })
}

pub fn register(c: &mut Container) {
    c.provide(provide_transactions_repository_ro);
    c.provide(provide_transactions_repository_wr);
    c.provide(provide_budgeting_service);
}
