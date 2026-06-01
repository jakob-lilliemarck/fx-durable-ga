use crate::infrastructure::db;
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

    pub async fn get_noise_diagnostic_config(
        &self,
        optimization_type_name: &str,
    ) -> Result<super::models::NoiseDiagnosticConfig, super::Error> {
        super::queries::get_noise_diagnostic_config(&self.ro.pool, optimization_type_name).await
    }

    pub async fn get_noise_diagnostic_runs(
        &self,
        noise_diagnostic_config_id: &Uuid,
    ) -> Result<Vec<Uuid>, super::Error> {
        super::queries::get_noise_diagnostic_runs(&self.ro.pool, noise_diagnostic_config_id).await
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

    pub async fn new_noise_diagnostic_config(
        &mut self,
        optimization_type_name: &str,
        probe_population_size: i32,
        probe_min_evaluations: i32,
        probe_max_evaluations: i32,
        budget_id: &Uuid,
    ) -> Result<super::models::NoiseDiagnosticConfig, super::Error> {
        super::queries::store_noise_diagnostic_config(
            &mut **self.tx,
            optimization_type_name,
            probe_population_size,
            probe_min_evaluations,
            probe_max_evaluations,
            budget_id,
        )
        .await
    }

    pub async fn store_noise_diagnostic_run(
        &mut self,
        id: &Uuid,
        noise_diagnostic_config_id: &Uuid,
    ) -> Result<Uuid, super::Error> {
        super::queries::store_noise_diagnostic_run(&mut **self.tx, &id, &noise_diagnostic_config_id)
            .await
    }
}
