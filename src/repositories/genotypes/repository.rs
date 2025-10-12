use crate::repositories::chainable::{Chain, ToTx};
use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use sqlx::{PgExecutor, PgPool, PgTransaction, prelude::FromRow};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Tx error: {0}")]
    Tx(anyhow::Error),
}

pub struct Repository {
    pool: PgPool,
}

pub struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub(crate) async fn new_genotype(&self, genotype: Genotype) -> Result<Genotype, Error> {
        new_genotype(&self.pool, genotype).await
    }

    pub(crate) async fn new_genotypes(
        &self,
        genotypes: Vec<Genotype>,
    ) -> Result<Vec<Genotype>, Error> {
        new_genotypes(&self.pool, genotypes).await
    }

    pub(crate) async fn get_genotype(&self, id: Uuid) -> Result<Genotype, Error> {
        get_genotype(&self.pool, id).await
    }

    pub(crate) async fn set_fitness(&self, id: Uuid, fitness: f64) -> Result<(), Error> {
        set_fitness(&self.pool, id, fitness).await
    }
}

impl<'tx> TxRepository<'tx> {
    pub fn new(tx: PgTransaction<'tx>) -> Self {
        Self { tx }
    }

    pub(crate) async fn new_genotype(&mut self, genotype: Genotype) -> Result<Genotype, Error> {
        new_genotype(&mut *self.tx, genotype).await
    }

    pub(crate) async fn new_genotypes(
        &mut self,
        genotypes: Vec<Genotype>,
    ) -> Result<Vec<Genotype>, Error> {
        new_genotypes(&mut *self.tx, genotypes).await
    }

    pub(crate) async fn get_genotype(&mut self, id: Uuid) -> Result<Genotype, Error> {
        get_genotype(&mut *self.tx, id).await
    }

    pub(crate) async fn set_fitness(&mut self, id: Uuid, fitness: f64) -> Result<(), Error> {
        set_fitness(&mut *self.tx, id, fitness).await
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}

impl<'tx> Chain<'tx> for Repository {
    type TxType = TxRepository<'tx>;
    type TxError = Error;

    fn chain<F, R, T>(&'tx self, f: F) -> BoxFuture<'tx, Result<T, Self::TxError>>
    where
        R: ToTx<'tx>,
        F: FnOnce(Self::TxType) -> BoxFuture<'tx, Result<(R, T), anyhow::Error>>
            + Send
            + Sync
            + 'tx,
        T: Send + Sync + 'tx,
    {
        Box::pin(async move {
            let pool = self.pool.clone();
            let tx = pool.begin().await?;

            let (tx, ret) = f(TxRepository { tx }).await.map_err(|err| Error::Tx(err))?;

            let tx: PgTransaction<'_> = tx.tx();
            tx.commit()
                .await
                .map_err(|err| Error::Tx(anyhow::Error::new(err)))?;

            Ok(ret)
        })
    }
}

pub type Gene = i64;

#[derive(Debug, FromRow)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct Genotype {
    pub(crate) id: Uuid,
    pub(crate) generated_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) genome: Vec<Gene>,
    pub(crate) request_id: Uuid,
    pub(crate) fitness: Option<f64>,
}

impl Genotype {
    pub fn new(type_name: &str, type_hash: i32, genome: Vec<Gene>, request_id: Uuid) -> Self {
        Self {
            id: Uuid::now_v7(),
            generated_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            genome,
            request_id,
            fitness: None,
        }
    }
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
                request_id,
                fitness;
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

pub async fn new_genotypes<'tx, E: PgExecutor<'tx>>(
    tx: E,
    genotypes: Vec<Genotype>,
) -> Result<Vec<Genotype>, Error> {
    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.genotypes (
            id,
            generated_at,
            type_name,
            type_hash,
            genome,
            request_id
        ) VALUES ",
    );

    let mut first = true;
    // Stream serialization: serialize each event as needed rather than
    // pre-allocating all payloads, reducing peak memory usage
    for g in genotypes {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        query_builder
            .push("(")
            .push_bind(g.id)
            .push(", ")
            .push_bind(g.generated_at)
            .push(", ")
            .push_bind(g.type_name)
            .push(", ")
            .push_bind(g.type_hash)
            .push(", ")
            .push_bind(g.genome)
            .push(", ")
            .push_bind(g.request_id)
            .push(")");
    }
    query_builder.push(" RETURNING id, generated_at, type_name, type_hash, genome, request_id");

    let genotypes = query_builder
        .build_query_as::<Genotype>()
        .fetch_all(tx)
        .await?;

    Ok(genotypes)
}

pub async fn set_fitness<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: Uuid,
    fitness: f64,
) -> Result<(), Error> {
    sqlx::query!(
        r#"
            UPDATE fx_durable_ga.genotypes
            SET fitness = $2
            WHERE id = $1 AND fitness IS NULL
            RETURNING id;
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
                request_id,
                fitness
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
    use chrono::SubsecRound;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id);
        let genotype_clone = genotype.clone();

        let inserted = repository.new_genotype(genotype).await?;

        assert_eq!(genotype_clone.id, inserted.id);
        assert_eq!(
            genotype_clone.generated_at.trunc_subsecs(6),
            inserted.generated_at
        );
        assert_eq!(genotype_clone.type_name, inserted.type_name);
        assert_eq!(genotype_clone.type_hash, inserted.type_hash);
        assert_eq!(genotype_clone.genome, inserted.genome);
        assert_eq!(genotype_clone.request_id, inserted.request_id);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id);
        let genotype_clone = genotype.clone();

        let _ = repository.new_genotype(genotype).await?;
        let result = repository.new_genotype(genotype_clone).await;

        assert!(result.is_err());
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_updates_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id);

        let inserted = repository.new_genotype(genotype).await?;
        repository.set_fitness(inserted.id, 0.9).await?;

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_if_fitness_is_already_set(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id);

        let inserted = repository.new_genotype(genotype).await?;
        repository.set_fitness(inserted.id, 0.9).await?;
        let result = repository.set_fitness(inserted.id, 0.8).await;

        assert!(result.is_err());
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let genotype = Genotype {
            id: Uuid::now_v7(),
            generated_at: Utc::now().trunc_subsecs(6),
            type_name: "test".to_string(),
            type_hash: 1,
            genome: vec![1, 2, 3],
            request_id: Uuid::now_v7(),
            fitness: None,
        };
        let genotype_id = genotype.id;

        let _ = repository.new_genotype(genotype).await?;
        let selected = repository.get_genotype(genotype_id).await?;

        assert_eq!(genotype_id, selected.id);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);
        let non_existent_id = Uuid::now_v7();

        let result = repository.get_genotype(non_existent_id).await;

        assert!(result.is_err());
        Ok(())
    }
}
