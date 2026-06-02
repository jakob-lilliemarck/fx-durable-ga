use super::Error;
use super::queries::SearchRequestsFilter;
use crate::infrastructure::db;
use crate::services::optimization::Request;
use sqlx::PgTransaction;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Read {
    ro: db::ReadPool,
}

#[derive(Debug, Clone)]
pub struct Write {
    wr: db::WritePool,
}

pub struct WriteTx<'tx> {
    tx: &'tx mut PgTransaction<'static>,
}

impl db::Tx for Write {
    type Error = super::Error;

    fn tx(self) -> db::TxFut<Self::Error> {
        let pool = self.wr.pool.clone();
        Box::pin(async move {
            let tx = pool.begin().await?;
            Ok(tx)
        })
    }
}

impl Read {
    pub fn new(ro: db::ReadPool) -> Self {
        Self { ro }
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn search_requests(
        &self,
        filter: &SearchRequestsFilter,
        limit: i64,
    ) -> Result<Vec<Request>, Error> {
        super::queries::search_requests(&self.ro.pool, filter, limit).await
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %id))]
    pub(crate) async fn get_request(&self, id: Uuid) -> Result<Request, Error> {
        super::queries::get_request(&self.ro.pool, &id).await
    }
}

impl Write {
    pub fn new(wr: db::WritePool) -> Self {
        Self { wr }
    }
}

impl<'tx> WriteTx<'tx> {
    #[instrument(level = "debug", skip(tx))]
    pub fn new(tx: &'tx mut PgTransaction<'static>) -> Self {
        Self { tx }
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request.id, type_name = %request.type_name, goal = ?request.goal))]
    pub(crate) async fn new_request(&mut self, request: Request) -> Result<Request, Error> {
        super::queries::store_request(&mut **self.tx, request).await
    }
}
