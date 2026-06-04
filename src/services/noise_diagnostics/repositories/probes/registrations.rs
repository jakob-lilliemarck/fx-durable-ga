use crate::infrastructure::{
    db,
    di::{Container, ProviderResult},
};
use futures::future::BoxFuture;

pub fn provide_probes_repository_ro(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Read>> {
    Box::pin(async {
        let ro = c.get::<db::ReadPool>().await?;
        let repository = super::Read::new(ro);
        Ok(repository)
    })
}

pub fn provide_probes_repository_wr(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Write>> {
    Box::pin(async {
        let wr = c.get::<db::WritePool>().await?;
        let repository = super::Write::new(wr);
        Ok(repository)
    })
}
