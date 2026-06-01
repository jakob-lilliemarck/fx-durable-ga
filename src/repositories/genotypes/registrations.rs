use crate::infrastructure::{
    db,
    di::{Container, ProviderResult},
};
use futures::future::BoxFuture;

pub fn provide_genotypes_repository_ro(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Read>> {
    Box::pin(async {
        let ro = c.get::<db::ReadPool>().await?;
        let repository = super::Read::new(ro);
        Ok(repository)
    })
}

pub fn provide_genotypes_repository_wr(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Write>> {
    Box::pin(async {
        let wr = c.get::<db::WritePool>().await?;
        let repository = super::Write::new(wr);
        Ok(repository)
    })
}

pub fn register(c: &mut Container) {
    c.provide(provide_genotypes_repository_ro);
    c.provide(provide_genotypes_repository_wr);
}
