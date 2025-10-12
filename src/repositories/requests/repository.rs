use serde::{Deserialize, Serialize};
use sqlx::{
    PgExecutor, PgPool, PgTransaction,
    types::chrono::{DateTime, Utc},
};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
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

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub enum Strategy {
    Rolling {
        max_evaluations: u32,
        population_size: u32,
        selection_interval: u32,
    },
    Generational {
        max_generations: u32,
        population_size: u32,
    },
}

/// A request for an optimization
#[derive(Debug)]
struct DbRequest {
    id: Uuid,
    requested_at: DateTime<Utc>,
    type_name: String,
    type_hash: i32,
    goal: FitnessGoal,
    threshold: f64,
    strategy: serde_json::Value,
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub struct Request {
    id: Uuid,
    requested_at: DateTime<Utc>,
    type_name: String,
    type_hash: i32,
    goal: FitnessGoal,
    threshold: f64,
    strategy: Strategy,
}

impl Request {
    pub fn new(
        name: &str,
        hash: i32,
        goal: FitnessGoal,
        threshold: f64,
        strategy: Strategy,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            requested_at: Utc::now(),
            type_name: name.to_string(),
            type_hash: hash,
            goal,
            threshold,
            strategy,
        }
    }
}

impl TryFrom<Request> for DbRequest {
    type Error = Error;

    fn try_from(request: Request) -> Result<Self, Self::Error> {
        let strategy_json = serde_json::to_value(request.strategy)?;

        Ok(DbRequest {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            type_hash: request.type_hash,
            goal: request.goal,
            threshold: request.threshold,
            strategy: strategy_json,
        })
    }
}

impl TryFrom<DbRequest> for Request {
    type Error = Error;

    fn try_from(request: DbRequest) -> Result<Self, Self::Error> {
        let strategy_json = serde_json::from_value(request.strategy)?;

        Ok(Request {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            type_hash: request.type_hash,
            goal: request.goal,
            threshold: request.threshold,
            strategy: strategy_json,
        })
    }
}

pub async fn new_request<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request: Request,
) -> Result<Request, Error> {
    let db_request: DbRequest = request.try_into()?;
    let db_request = sqlx::query_as!(
        DbRequest,
        r#"
            INSERT INTO fx_durable_ga.requests (
                id,
                requested_at,
                type_name,
                type_hash,
                goal,
                threshold,
                strategy
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING
                id,
                requested_at,
                type_name,
                type_hash,
                goal "goal:FitnessGoal",
                threshold,
                strategy
            "#,
        db_request.id,
        db_request.requested_at,
        db_request.type_name,
        db_request.type_hash,
        db_request.goal as FitnessGoal,
        db_request.threshold,
        db_request.strategy
    )
    .fetch_one(tx)
    .await?;

    let request: Request = db_request.try_into()?;
    Ok(request)
}

pub async fn get_request<'tx, E: PgExecutor<'tx>>(tx: E, id: Uuid) -> Result<Request, Error> {
    let db_request = sqlx::query_as!(
        DbRequest,
        r#"
        SELECT
            id,
            requested_at,
            type_name,
            type_hash,
            goal "goal!:FitnessGoal",
            threshold,
            strategy
        FROM fx_durable_ga.requests
        WHERE id = $1
        "#,
        id
    )
    .fetch_one(tx)
    .await?;

    let request: Request = db_request.try_into()?;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::SubsecRound;
    use crate::migrations::run_migrations;

    async fn setup_test_db(pool: &PgPool) -> Result<(), sqlx::Error> {
        // Drop schema if it exists
        sqlx::query!("DROP SCHEMA IF EXISTS fx_durable_ga CASCADE")
            .execute(pool)
            .await?;

        // Create schema
        sqlx::query!("CREATE SCHEMA fx_durable_ga")
            .execute(pool)
            .await?;

        // Run migrations
        run_migrations(pool).await?;

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        setup_test_db(&pool).await?;
        let repository = Repository::new(pool);

        let request = Request::new(
            "test",
            1,
            FitnessGoal::Maximize,
            0.9,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
        );
        let request_clone = request.clone();

        let inserted = repository.new_request(request).await?;

        assert_eq!(request_clone.id, inserted.id);
        assert_eq!(
            request_clone.requested_at.trunc_subsecs(6),
            inserted.requested_at
        );
        assert_eq!(request_clone.type_name, inserted.type_name);
        assert_eq!(request_clone.type_hash, inserted.type_hash);
        assert_eq!(request_clone.goal, inserted.goal);
        assert_eq!(request_clone.threshold, inserted.threshold);
        assert_eq!(request_clone.strategy, inserted.strategy);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        setup_test_db(&pool).await?;
        let repository = Repository::new(pool);

        let request = Request::new(
            "test",
            1,
            FitnessGoal::Maximize,
            0.9,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
        );
        let request_clone = request.clone();

        let _ = repository.new_request(request).await?;
        let inserted = repository.new_request(request_clone).await;

        assert!(inserted.is_err());

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        setup_test_db(&pool).await?;
        let repository = Repository::new(pool);

        let request = Request::new(
            "test",
            1,
            FitnessGoal::Maximize,
            0.9,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
        );
        let request_id = request.id;

        let _ = repository.new_request(request).await?;
        let selected = repository.get_request(request_id).await?;

        assert_eq!(request_id, selected.id);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        setup_test_db(&pool).await?;
        let repository = Repository::new(pool);

        let request = Request::new(
            "test",
            1,
            FitnessGoal::Maximize,
            0.9,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
        );
        let request_id = request.id;

        let selected = repository.get_request(request_id).await;

        assert!(selected.is_err());
        Ok(())
    }
}
