use futures::future::BoxFuture;
use sqlx::PgTransaction;

// ============================================================
// Implementations on the Repository
// ============================================================
pub trait TxType<'tx> {
    type TxType: ToTx<'tx>;
    type TxError;
}

pub trait Chain<'tx>: TxType<'tx> {
    fn chain<F, R, T>(&'tx self, f: F) -> BoxFuture<'tx, Result<T, Self::TxError>>
    where
        R: ToTx<'tx>,
        F: FnOnce(Self::TxType) -> BoxFuture<'tx, Result<(R, T), anyhow::Error>>
            + Send
            + Sync
            + 'tx,
        T: Send + Sync + 'tx;
}

pub trait FromOther<'tx>: TxType<'tx> {
    fn from_other(&self, other: impl ToTx<'tx>) -> Self::TxType;
}

// ============================================================
// Implementations on the TxType
// ============================================================
pub trait ToTx<'tx>: Send + Sync {
    fn tx(self) -> PgTransaction<'tx>;
}

pub trait FromTx<'tx> {
    fn from_tx(other: impl ToTx<'tx>) -> Self;
}

// Implementations on foreign types
impl<'tx> FromTx<'tx> for fx_event_bus::Publisher<'tx> {
    fn from_tx(other: impl ToTx<'tx>) -> Self {
        let tx: PgTransaction<'_> = other.tx();
        fx_event_bus::Publisher::new(tx)
    }
}
impl<'tx> ToTx<'tx> for fx_event_bus::Publisher<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.into()
    }
}
