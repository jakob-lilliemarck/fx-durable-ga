use sqlx::{
    PgExecutor, PgPool, PgTransaction,
    types::chrono::{DateTime, Utc},
};
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
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn new_request(&self, request: Request) -> Result<Request, Error> {
        new_request(&self.pool, request).await
    }

    pub async fn get_request(&self, id: Uuid) -> Result<Request, Error> {
        get_request(&self.pool, id).await
    }
}

impl<'tx> TxRepository<'tx> {
    pub fn new(tx: PgTransaction<'tx>) -> Self {
        Self { tx }
    }

    pub async fn new_request(&mut self, request: Request) -> Result<Request, Error> {
        new_request(&mut *self.tx, request).await
    }

    pub async fn get_request(&mut self, id: Uuid) -> Result<Request, Error> {
        get_request(&mut *self.tx, id).await
    }
}

#[derive(sqlx::Type, Debug, Clone, PartialEq)]
#[sqlx(type_name = "fx_durable_ga.fitness_goal", rename_all = "lowercase")]
pub enum FitnessGoal {
    Minimize,
    Maximize,
    Exact,
}

/// A request for an optimization
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// Identifier
    id: Uuid,
    /// The time this request was received
    requested_at: DateTime<Utc>,
    /// A human readable name of the type to optimize
    name: String,
    /// A hash value of the name
    hash: i32,
    /// An indicator if we're maximizing, minimizing or looking for an exact fitness value
    goal: FitnessGoal,
    /// The threshold value at which point to complete
    threshold: f64,
    /// The maximum number of generations
    max_generations: i64,
    /// The population size of each generation
    population_size: i64,
}

impl Request {
    pub fn new(
        name: &str,
        hash: i32,
        goal: FitnessGoal,
        threshold: f64,
        max_generations: u32,
        population_size: u32,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            requested_at: Utc::now(),
            name: name.to_string(),
            hash,
            goal,
            threshold,
            max_generations: max_generations as i64,
            population_size: population_size as i64,
        }
    }
}

pub async fn new_request<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request: Request,
) -> Result<Request, Error> {
    let request = sqlx::query_as!(
        Request,
        r#"
            INSERT INTO fx_durable_ga.requests (
                id,
                requested_at,
                name,
                hash,
                goal,
                threshold,
                max_generations,
                population_size
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING
                id,
                requested_at,
                name,
                hash,
                goal "goal:FitnessGoal",
                threshold,
                max_generations,
                population_size
            "#,
        request.id,
        request.requested_at,
        request.name,
        request.hash,
        request.goal as FitnessGoal,
        request.threshold,
        request.max_generations,
        request.population_size
    )
    .fetch_one(tx)
    .await?;

    Ok(request)
}

pub async fn get_request<'tx, E: PgExecutor<'tx>>(tx: E, id: Uuid) -> Result<Request, Error> {
    let request = sqlx::query_as!(
        Request,
        r#"
        SELECT
            id,
            requested_at,
            name,
            hash,
            goal "goal!:FitnessGoal",
            threshold,
            max_generations,
            population_size
        FROM fx_durable_ga.requests
        WHERE id = $1
        "#,
        id
    )
    .fetch_one(tx)
    .await?;

    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request = Request::new("test", 1, FitnessGoal::Maximize, 0.9, 100, 10);
        let request_id = request.id;

        let inserted = repository.new_request(request).await?;

        assert_eq!(request_id, inserted.id);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request = Request::new("test", 1, FitnessGoal::Maximize, 0.9, 100, 10);
        let request_clone = request.clone();

        let _ = repository.new_request(request).await?;
        let inserted = repository.new_request(request_clone).await;

        assert!(inserted.is_err());

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request = Request::new("test", 1, FitnessGoal::Maximize, 0.9, 100, 10);
        let request_id = request.id;

        let _ = repository.new_request(request).await?;
        let selected = repository.get_request(request_id).await?;

        assert_eq!(request_id, selected.id);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let request = Request::new("test", 1, FitnessGoal::Maximize, 0.9, 100, 10);
        let request_id = request.id;

        let selected = repository.get_request(request_id).await;

        assert!(selected.is_err());
        Ok(())
    }
}
