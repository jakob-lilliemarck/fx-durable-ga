use crate::infrastructure::db;
use crate::infrastructure::di::{Container, ProviderResult};
use futures::future::BoxFuture;

pub fn provide_evaluations_repository_ro(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Read>> {
    Box::pin(async {
        let ro = c.get::<db::ReadPool>().await?;
        Ok(super::Read::new(ro))
    })
}

pub fn provide_evaluations_repository_wr(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Write>> {
    Box::pin(async {
        let wr = c.get::<db::WritePool>().await?;
        Ok(super::Write::new(wr))
    })
}
