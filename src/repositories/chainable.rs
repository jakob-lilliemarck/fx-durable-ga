use futures::future::BoxFuture;
use sqlx::PgTransaction;

// ============================================================
// Implementations on the Repository
// ============================================================

/// Defines the transaction type and error type for a repository.
pub trait TxType<'tx> {
    type TxType: ToTx<'tx>;
    type TxError;
}

/// Provides transactional chaining capabilities for repositories.
///
/// Allows executing operations within a database transaction that can be committed or rolled back.
pub trait Chain<'tx>: TxType<'tx> {
    /// Executes a function within a database transaction.
    ///
    /// The transaction is automatically committed if the function succeeds, or rolled back on error.
    fn chain<F, R, T>(&'tx self, f: F) -> BoxFuture<'tx, Result<T, Self::TxError>>
    where
        R: ToTx<'tx>,
        F: FnOnce(Self::TxType) -> BoxFuture<'tx, Result<(R, T), anyhow::Error>>
            + Send
            + Sync
            + 'tx,
        T: Send + Sync + 'tx;
}

/// Creates a transactional repository from another transactional object.
pub trait FromOther<'tx>: TxType<'tx> {
    /// Creates a new transaction type from another transaction-capable object.
    fn from_other(&self, other: impl ToTx<'tx>) -> Self::TxType;
}

// ============================================================
// Implementations on the TxType
// ============================================================

/// Converts a type into a PostgreSQL database transaction.
pub trait ToTx<'tx>: Send + Sync {
    /// Consumes self and returns the underlying database transaction.
    fn tx(self) -> PgTransaction<'tx>;
}

/// Creates a type from a transaction-capable object.
pub trait FromTx<'tx> {
    /// Creates a new instance from another transaction-capable object.
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

impl<'tx> ToTx<'tx> for PgTransaction<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self
    }
}
