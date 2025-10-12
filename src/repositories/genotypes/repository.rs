use crate::gene::Gene;
use chrono::{DateTime, Utc};
use sqlx::{PgExecutor, PgPool, PgTransaction};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
}

pub struct Repository {
    pool: PgPool,
}

pub struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl Repository {
    pub(crate) async fn new_genotype(&self, genotype: Genotype) -> Result<Genotype, Error> {
        new_genotype(&self.pool, genotype).await
    }

    pub(crate) async fn get_genotype(&self, id: Uuid) -> Result<Genotype, Error> {
        get_genotype(&self.pool, id).await
    }

    pub(crate) async fn set_fitness(&self, id: Uuid, fitness: f64) -> Result<(), Error> {
        set_fitness(&self.pool, id, fitness).await
    }
}

impl<'tx> TxRepository<'tx> {
    pub(crate) async fn new_genotype(&mut self, genotype: Genotype) -> Result<Genotype, Error> {
        new_genotype(&mut *self.tx, genotype).await
    }

    pub(crate) async fn get_genotype(&mut self, id: Uuid) -> Result<Genotype, Error> {
        get_genotype(&mut *self.tx, id).await
    }

    pub(crate) async fn set_fitness(&mut self, id: Uuid, fitness: f64) -> Result<(), Error> {
        set_fitness(&mut *self.tx, id, fitness).await
    }
}

pub struct Genotype {
    id: Uuid,
    generated_at: DateTime<Utc>,
    type_name: String,
    type_hash: i32,
    genome: Vec<Gene>,
    request_id: Uuid,
}

pub async fn new_genotype<'tx, E: PgExecutor<'tx>>(
    tx: E,
    genotype: Genotype,
) -> Result<Genotype, Error> {
    let genotype = sqlx::query_as!(
        Genotype,
        r#"
            INSERT INTO fx_durable_ga.genotypes (
                id,
                generated_at,
                type_name,
                type_hash,
                genome,
                request_id
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING
                id,
                generated_at,
                type_name,
                type_hash,
                genome,
                request_id;
            "#,
        genotype.id,
        genotype.generated_at,
        genotype.type_name,
        genotype.type_hash,
        &genotype.genome,
        genotype.request_id
    )
    .fetch_one(tx)
    .await?;

    Ok(genotype)
}

pub async fn set_fitness<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: Uuid,
    fitness: f64,
) -> Result<(), Error> {
    sqlx::query_as!(
        Genotype,
        r#"
            UPDATE fx_durable_ga.genotypes
            SET fitness = $2
            WHERE id = $1 AND fitness IS NULL;
        "#,
        id,
        fitness
    )
    .fetch_one(tx)
    .await?;

    Ok(())
}

pub async fn get_genotype<'tx, E: PgExecutor<'tx>>(tx: E, id: Uuid) -> Result<Genotype, Error> {
    let genotype = sqlx::query_as!(
        Genotype,
        r#"
            SELECT
                id,
                generated_at,
                type_name,
                type_hash,
                genome,
                request_id
            FROM fx_durable_ga.genotypes
            WHERE id = $1;
        "#,
        id
    )
    .fetch_one(tx)
    .await?;

    Ok(genotype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_updates_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_if_fitness_is_already_set(pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }
}
