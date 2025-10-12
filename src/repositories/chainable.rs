use futures::future::BoxFuture;
use sqlx::PgTransaction;

pub trait Chain<'tx> {
    type TxType: ToTx<'tx>;
    type TxError;

    fn chain<F, R, T>(&'tx self, f: F) -> BoxFuture<'tx, Result<T, Self::TxError>>
    where
        R: ToTx<'tx>,
        F: FnOnce(Self::TxType) -> BoxFuture<'tx, Result<(R, T), anyhow::Error>>
            + Send
            + Sync
            + 'tx,
        T: Send + Sync + 'tx;
}

pub trait ToTx<'tx>: Send + Sync {
    fn tx(self) -> PgTransaction<'tx>;
}

pub trait FromTxType<'tx> {
    fn from_other(other: impl ToTx<'tx>) -> Self;
}

// Implement for event bus
impl<'tx> FromTxType<'tx> for fx_event_bus::Publisher<'tx> {
    fn from_other(other: impl ToTx<'tx>) -> Self {
        let tx: PgTransaction<'_> = other.tx();
        fx_event_bus::Publisher::new(tx)
    }
}
impl<'tx> ToTx<'tx> for fx_event_bus::Publisher<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.into()
    }
}
