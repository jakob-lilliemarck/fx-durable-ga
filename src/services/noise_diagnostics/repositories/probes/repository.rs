use super::models::{NoiseProbe, SearchNoiseProbesFilter};
use crate::infrastructure::db;
use sqlx::PgTransaction;
use tracing::instrument;

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

    pub async fn search_noise_probes(
        &self,
        filter: &SearchNoiseProbesFilter,
    ) -> Result<Vec<NoiseProbe>, super::Error> {
        super::queries::search_noise_probes(&self.ro.pool, filter).await
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

    pub fn tx(&mut self) -> &mut PgTransaction<'static> {
        self.tx
    }

    pub async fn store_noise_probe(
        &mut self,
        probe: &NoiseProbe,
    ) -> Result<NoiseProbe, super::Error> {
        super::queries::store_noise_probe(&mut **self.tx, probe).await
    }
}
