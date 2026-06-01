use super::repository::EncoderCache;
use crate::infrastructure::{
    db,
    di::{Container, ProviderResult},
};
use futures::future::BoxFuture;
use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

pub fn provide_encoders_repository_ro(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Read>> {
    Box::pin(async {
        let ro = c.get::<db::ReadPool>().await?;

        let cache = Arc::new(RwLock::new(EncoderCache::new(
            Duration::from_secs(60 * 60),
            20,
        )));

        let encoders = super::Read::new(ro, cache);

        Ok(encoders)
    })
}

pub fn provide_encoders_repository_wr(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Write>> {
    Box::pin(async {
        let wr = c.get::<db::WritePool>().await?;

        let repository = super::Write::new(wr);

        Ok(repository)
    })
}
