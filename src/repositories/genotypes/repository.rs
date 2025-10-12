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

    pub(crate) async fn count_genotypes_in_latest_iteration(
        &self,
        filter: &Filter,
    ) -> Result<i64, Error> {
        count_genotypes_in_latest_iteration(&self.pool, filter).await
    }

    pub(crate) async fn search_genotypes_in_latest_generation(
        &self,
        limit: i64,
        filter: &Filter,
    ) -> Result<Vec<Genotype>, Error> {
        search_genotypes_in_latest_generation(&self.pool, limit, filter).await
    }

    pub(crate) async fn add_to_population(&mut self, pairs: &[(Uuid, Uuid)]) -> Result<(), Error> {
        add_to_population(&self.pool, &pairs).await
    }

    pub(crate) async fn remove_from_population(
        &mut self,
        request_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        remove_from_population(&self.pool, request_id, genotype_id).await
    }

    pub(crate) async fn get_population_count(&self, request_id: Uuid) -> Result<i64, Error> {
        get_population_count(&self.pool, request_id).await
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

    pub(crate) async fn count_genotypes_in_latest_iteration(
        &mut self,
        filter: &Filter,
    ) -> Result<i64, Error> {
        count_genotypes_in_latest_iteration(&mut *self.tx, filter).await
    }

    pub(crate) async fn search_genotypes_in_latest_generation(
        &mut self,
        limit: i64,
        filter: &Filter,
    ) -> Result<Vec<Genotype>, Error> {
        search_genotypes_in_latest_generation(&mut *self.tx, limit, filter).await
    }

    pub(crate) async fn add_to_population(&mut self, pairs: &[(Uuid, Uuid)]) -> Result<(), Error> {
        add_to_population(&mut *self.tx, pairs).await
    }

    pub(crate) async fn remove_from_population(
        &mut self,
        request_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        remove_from_population(&mut *self.tx, request_id, genotype_id).await
    }

    pub(crate) async fn get_population_count(&mut self, request_id: Uuid) -> Result<i64, Error> {
        get_population_count(&mut *self.tx, request_id).await
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
    pub(crate) generation_id: i32,
}

impl Genotype {
    pub fn new(
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

    pub fn fitness(&self) -> Option<f64> {
        self.fitness
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
                request_id,
                generation_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING
                id,
                generated_at,
                type_name,
                type_hash,
                genome,
                request_id,
                fitness,
                generation_id;
            "#,
        genotype.id,
        genotype.generated_at,
        genotype.type_name,
        genotype.type_hash,
        &genotype.genome,
        genotype.request_id,
        genotype.generation_id
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
                fitness,
                generation_id
            FROM fx_durable_ga.genotypes
            WHERE id = $1;
        "#,
        id
    )
    .fetch_one(tx)
    .await?;

    Ok(genotype)
}

pub struct Filter {
    ids: Option<Vec<Uuid>>,
    request_ids: Option<Vec<Uuid>>,
    has_fitness: Option<bool>,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            ids: None,
            has_fitness: None,
            request_ids: None,
        }
    }
}

impl Filter {
    pub fn with_fitness(mut self, has_fitness: bool) -> Self {
        self.has_fitness = Some(has_fitness);
        self
    }

    pub fn with_ids(mut self, ids: Vec<Uuid>) -> Self {
        self.ids = Some(ids);
        self
    }

    pub fn with_request_ids(mut self, request_ids: Vec<Uuid>) -> Self {
        self.request_ids = Some(request_ids);
        self
    }
}

pub async fn count_genotypes_in_latest_iteration<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &Filter,
) -> Result<i64, Error> {
    // Return i64, not Genotype - we're counting!
    let count = sqlx::query_scalar!(
        // No need to specify i64, query_scalar! infers it
        r#"
            WITH latest_generations AS (
                SELECT request_id, MAX(generation_id) as max_generation_id
                FROM fx_durable_ga.genotypes
                GROUP BY request_id
            )
            SELECT COUNT(*) "count!:i64"
            FROM fx_durable_ga.genotypes g
            JOIN latest_generations lg
                ON g.request_id = lg.request_id
                AND g.generation_id = lg.max_generation_id
            WHERE (
                $1::bool IS NULL OR
                CASE
                    WHEN $1 = true THEN g.fitness IS NOT NULL
                    ELSE g.fitness IS NULL
                END
            )
            AND (
                $2::uuid[] IS NULL OR g.id = ANY($2)
            )
            AND (
                $3::uuid[] IS NULL OR g.request_id = ANY($3)
            );
        "#,
        filter.has_fitness,
        filter.ids.as_deref(),
        filter.request_ids.as_deref(),
    )
    .fetch_one(tx)
    .await?;

    Ok(count)
}

pub async fn search_genotypes_in_latest_generation<'tx, E: PgExecutor<'tx>>(
    tx: E,
    limit: i64,
    filter: &Filter,
) -> Result<Vec<Genotype>, Error> {
    // Return i64, not Genotype - we're counting!
    let genotypes = sqlx::query_as!(
        Genotype,
        r#"
        WITH latest_generations AS (
            SELECT request_id, MAX(generation_id) as max_generation_id
            FROM fx_durable_ga.genotypes
            GROUP BY request_id
        )
        SELECT
            g.id,
            g.generated_at,
            g.type_name,
            g.type_hash,
            g.genome,
            g.request_id,
            g.fitness,
            g.generation_id
        FROM fx_durable_ga.genotypes g
        JOIN latest_generations lg
            ON g.request_id = lg.request_id
            AND g.generation_id = lg.max_generation_id
        WHERE (
            $1::bool IS NULL OR
            CASE
                WHEN $1 = true THEN g.fitness IS NOT NULL
                ELSE g.fitness IS NULL
            END
        )
        AND (
            $2::uuid[] IS NULL OR g.id = ANY($2)
        )
        AND (
            $3::uuid[] IS NULL OR g.request_id = ANY($3)
        )
        ORDER BY g.fitness DESC NULLS LAST
        LIMIT $4;
        "#,
        filter.has_fitness,
        filter.ids.as_deref(),
        filter.request_ids.as_deref(),
        limit
    )
    .fetch_all(tx)
    .await?;

    Ok(genotypes)
}

pub async fn add_to_population<'tx, E: PgExecutor<'tx>>(
    tx: E,
    pairs: &[(Uuid, Uuid)],
) -> Result<(), Error> {
    // paris is (request_id, genotype_id)
    todo!()
}

pub async fn remove_from_population<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: Uuid,
    genotype_id: Uuid,
) -> Result<(), Error> {
    todo!()
}

pub async fn get_population_count<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: Uuid,
) -> Result<i64, Error> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[sqlx::test(migrations = "./migrations")]
    async fn it_counts_some_genotypes(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);
        let req_id_1 = Uuid::now_v7();
        let req_id_2 = Uuid::now_v7();

        assert_eq!(
            repository
                .count_genotypes_in_latest_iteration(&Filter::default())
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
                .count_genotypes_in_latest_iteration(&Filter::default())
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
                .count_genotypes_in_latest_iteration(&Filter::default())
                .await?,
            2
        );
        assert_eq!(
            repository
                .count_genotypes_in_latest_iteration(&Filter::default().with_fitness(false))
                .await?,
            1
        );
        assert_eq!(
            repository
                .count_genotypes_in_latest_iteration(&Filter::default().with_fitness(true))
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
                .count_genotypes_in_latest_iteration(
                    &Filter::default().with_request_ids(vec![req_id_1])
                )
                .await?,
            2
        );
        assert_eq!(
            repository
                .count_genotypes_in_latest_iteration(
                    &Filter::default().with_request_ids(vec![req_id_2])
                )
                .await?,
            1
        );
        assert_eq!(
            repository
                .count_genotypes_in_latest_iteration(
                    &Filter::default().with_request_ids(vec![req_id_1, req_id_2])
                )
                .await?,
            3
        );
        assert_eq!(
            repository
                .count_genotypes_in_latest_iteration(
                    &Filter::default()
                        .with_request_ids(vec![req_id_1, req_id_2])
                        .with_fitness(false)
                )
                .await?,
            2
        );
        assert_eq!(
            repository
                .count_genotypes_in_latest_iteration(
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
