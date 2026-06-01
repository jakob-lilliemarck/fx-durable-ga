use futures::future::BoxFuture;
use sqlx::{PgPool, PgTransaction};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("transaction failed")]
    Transaction(#[from] sqlx::Error),
}

pub type TxFut<E> = BoxFuture<'static, Result<PgTransaction<'static>, E>>;

pub trait Tx {
    type Error;

    fn tx(self) -> TxFut<Self::Error>;
}

impl Tx for fx_mq_jobs::Publisher<PgTransaction<'static>> {
    type Error = sqlx::Error;

    fn tx(self) -> TxFut<Self::Error> {
        Box::pin(async move { Ok(self.into()) })
    }
}

pub async fn begin<T, F, R>(other: T, f: F) -> Result<R, anyhow::Error>
where
    T: Tx,
    T::Error: Into<anyhow::Error>,
    F: for<'tx> FnOnce(&'tx mut PgTransaction<'static>) -> BoxFuture<'tx, Result<R, anyhow::Error>>,
{
    let mut tx = other.tx().await.map_err(Into::into)?;
    let result = f(&mut tx).await?;
    tx.commit().await?;
    Ok(result)
}

#[derive(Debug, Clone)]
pub struct WritePool {
    pub pool: PgPool,
}

#[derive(Debug, Clone)]
pub struct ReadPool {
    pub pool: PgPool,
}
