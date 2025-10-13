use crate::repositories::chainable::{Chain, ToTx, TxType};
use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use sqlx::{PgPool, PgTransaction, prelude::FromRow};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Tx error: {0}")]
    Tx(anyhow::Error),
}

pub(crate) type Gene = i64;

#[derive(Debug, Clone, FromRow)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Genotype {
    pub(crate) id: Uuid,
    pub(crate) generated_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) genome: Vec<Gene>,
    pub(crate) request_id: Uuid,
    pub(crate) fitness: Option<f64>,
    pub(crate) generation_id: i32,
}

impl Genotype {
    pub(crate) fn new(
        type_name: &str,
        type_hash: i32,
        genome: Vec<Gene>,
        request_id: Uuid,
        generation_id: i32,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            generated_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            genome,
            request_id,
            fitness: None,
            generation_id,
        }
    }

    /// Panics if the genotype does not have fitness
    pub(crate) fn must_fitness(&self) -> f64 {
        self.fitness.expect("Genotype must have a fitness value")
    }
}

pub struct Repository {
    pool: PgPool,
}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub(crate) async fn new_genotype(&self, genotype: Genotype) -> Result<Genotype, Error> {
        super::queries::new_genotype(&self.pool, genotype).await
    }

    pub(crate) async fn new_genotypes(
        &self,
        genotypes: Vec<Genotype>,
    ) -> Result<Vec<Genotype>, Error> {
        super::queries::new_genotypes(&self.pool, genotypes).await
    }

    pub(crate) async fn get_genotype(&self, id: &Uuid) -> Result<Genotype, Error> {
        super::queries::get_genotype(&self.pool, id).await
    }

    pub(crate) async fn set_fitness(&self, id: Uuid, fitness: f64) -> Result<(), Error> {
        super::queries::set_fitness(&self.pool, id, fitness).await
    }

    pub(crate) async fn get_count_of_genotypes_in_latest_iteration(
        &self,
        filter: &super::queries::Filter,
    ) -> Result<i64, Error> {
        super::queries::count_genotypes_in_latest_iteration(&self.pool, filter).await
    }

    pub(crate) async fn search_genotypes_in_latest_generation(
        &self,
        limit: i64,
        order: super::queries::Order,
        filter: &super::queries::Filter,
    ) -> Result<Vec<Genotype>, Error> {
        super::queries::search_genotypes_in_latest_generation(&self.pool, limit, order, filter)
            .await
    }

    pub(crate) async fn get_generation_coun(&self, request_id: Uuid) -> Result<i32, Error> {
        super::queries::get_generation_count(&self.pool, request_id).await
    }
}

impl<'tx> TxType<'tx> for Repository {
    type TxType = TxRepository<'tx>;
    type TxError = Error;
}

impl<'tx> Chain<'tx> for Repository {
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

pub struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    pub(crate) async fn new_genotype(&mut self, genotype: Genotype) -> Result<Genotype, Error> {
        super::queries::new_genotype(&mut *self.tx, genotype).await
    }

    pub(crate) async fn new_genotypes(
        &mut self,
        genotypes: Vec<Genotype>,
    ) -> Result<Vec<Genotype>, Error> {
        super::queries::new_genotypes(&mut *self.tx, genotypes).await
    }

    pub(crate) async fn get_genotype(&mut self, id: &Uuid) -> Result<Genotype, Error> {
        super::queries::get_genotype(&mut *self.tx, id).await
    }

    pub(crate) async fn set_fitness(&mut self, id: Uuid, fitness: f64) -> Result<(), Error> {
        super::queries::set_fitness(&mut *self.tx, id, fitness).await
    }

    pub(crate) async fn count_genotypes_in_latest_iteration(
        &mut self,
        filter: &super::queries::Filter,
    ) -> Result<i64, Error> {
        super::queries::count_genotypes_in_latest_iteration(&mut *self.tx, filter).await
    }

    pub(crate) async fn search_genotypes_in_latest_generation(
        &mut self,
        limit: i64,
        order: super::queries::Order,
        filter: &super::queries::Filter,
    ) -> Result<Vec<Genotype>, Error> {
        super::queries::search_genotypes_in_latest_generation(&mut *self.tx, limit, order, filter)
            .await
    }

    pub(crate) async fn get_generation_count(&mut self, request_id: Uuid) -> Result<i32, Error> {
        super::queries::get_generation_count(&mut *self.tx, request_id).await
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repositories::genotypes::Filter;
    use chrono::SubsecRound;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id, 1);
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
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id, 1);
        let genotype_clone = genotype.clone();

        repository.new_genotype(genotype).await?;
        let result = repository.new_genotype(genotype_clone).await;

        assert!(result.is_err());
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_updates_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id, 1);

        let inserted = repository.new_genotype(genotype).await?;
        repository.set_fitness(inserted.id, 0.9).await?;

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_if_fitness_is_already_set(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id, 1);

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
            generation_id: 1,
        };
        let genotype_id = genotype.id;

        repository.new_genotype(genotype).await?;
        let selected = repository.get_genotype(&genotype_id).await?;

        assert_eq!(genotype_id, selected.id);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);
        let non_existent_id = Uuid::now_v7();

        let result = repository.get_genotype(&non_existent_id).await;

        assert!(result.is_err());
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_counts_some_genotypes(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);
        let req_id_1 = Uuid::now_v7();
        let req_id_2 = Uuid::now_v7();

        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(&Filter::default())
                .await?,
            0
        );

        // FIRST
        repository
            .new_genotype(Genotype {
                id: Uuid::now_v7(),
                generated_at: Utc::now().trunc_subsecs(6),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![1, 2, 3],
                request_id: req_id_1,
                fitness: None,
                generation_id: 1,
            })
            .await?;

        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(&Filter::default())
                .await?,
            1
        );

        // SECOND
        let genotype_id_2 = Uuid::now_v7();
        repository
            .new_genotype(Genotype {
                id: genotype_id_2,
                generated_at: Utc::now().trunc_subsecs(6),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![1, 2, 3],
                request_id: req_id_1,
                fitness: None,
                generation_id: 1,
            })
            .await?;
        repository.set_fitness(genotype_id_2, 0.123).await?;

        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(&Filter::default())
                .await?,
            2
        );
        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(&Filter::default().with_fitness(false))
                .await?,
            1
        );
        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(&Filter::default().with_fitness(true))
                .await?,
            1
        );

        // THIRD
        repository
            .new_genotype(Genotype {
                id: Uuid::now_v7(),
                generated_at: Utc::now().trunc_subsecs(6),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![1, 2, 3],
                request_id: req_id_2,
                fitness: None,
                generation_id: 1,
            })
            .await?;

        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(
                    &Filter::default().with_request_ids(vec![req_id_1])
                )
                .await?,
            2
        );
        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(
                    &Filter::default().with_request_ids(vec![req_id_2])
                )
                .await?,
            1
        );
        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(
                    &Filter::default().with_request_ids(vec![req_id_1, req_id_2])
                )
                .await?,
            3
        );
        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(
                    &Filter::default()
                        .with_request_ids(vec![req_id_1, req_id_2])
                        .with_fitness(false)
                )
                .await?,
            2
        );
        assert_eq!(
            repository
                .get_count_of_genotypes_in_latest_iteration(
                    &Filter::default()
                        .with_request_ids(vec![req_id_1, req_id_2])
                        .with_fitness(true)
                )
                .await?,
            1
        );
        Ok(())
    }
}
