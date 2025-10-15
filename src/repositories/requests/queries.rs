use super::Error;
use super::models::DbRequest;
use crate::models::Request;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx), fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash, goal = ?request.goal))]
pub(crate) async fn new_request<'tx, E: PgExecutor<'tx>>(
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
                strategy,
                mutagen,
                crossover
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING
                id,
                requested_at,
                type_name,
                type_hash,
                goal,
                strategy,
                mutagen,
                crossover
            "#,
        db_request.id,
        db_request.requested_at,
        db_request.type_name,
        db_request.type_hash,
        db_request.goal,
        db_request.strategy,
        db_request.mutagen,
        db_request.crossover
    )
    .fetch_one(tx)
    .await?;

    let request: Request = db_request.try_into()?;
    Ok(request)
}

#[cfg(test)]
mod new_request_tests {
    use super::*;
    use crate::models::FitnessGoal;
    use crate::models::{Crossover, Mutagen, Strategy};
    use chrono::SubsecRound;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let mutagen = Mutagen::constant(0.5, 0.1)?;
        let crossover = Crossover::uniform(0.5)?;
        let goal = FitnessGoal::maximize(0.9)?;

        let request = Request::new(
            "test",
            1,
            goal,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            mutagen,
            crossover,
        )?;
        let request_clone = request.clone();

        let inserted = new_request(&pool, request).await?;

        assert_eq!(request_clone.id, inserted.id);
        assert_eq!(
            request_clone.requested_at.trunc_subsecs(6),
            inserted.requested_at
        );
        assert_eq!(request_clone.type_name, inserted.type_name);
        assert_eq!(request_clone.type_hash, inserted.type_hash);
        assert_eq!(request_clone.goal, inserted.goal);
        assert_eq!(request_clone.strategy, inserted.strategy);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let mutagen = Mutagen::constant(0.5, 0.1)?;
        let crossover = Crossover::uniform(0.5)?;
        let goal = FitnessGoal::maximize(0.9)?;

        let request = Request::new(
            "test",
            1,
            goal,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            mutagen,
            crossover,
        )?;
        let request_clone = request.clone();

        new_request(&pool, request).await?;
        let inserted = new_request(&pool, request_clone).await;

        assert!(inserted.is_err());

        Ok(())
    }
}

#[instrument(level = "debug", skip(tx), fields(request_id = %id))]
pub(crate) async fn get_request<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: &Uuid,
) -> Result<Request, Error> {
    let db_request = sqlx::query_as!(
        DbRequest,
        r#"
        SELECT
            id,
            requested_at,
            type_name,
            type_hash,
            goal,
            strategy,
            mutagen,
            crossover
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
mod get_request_tests {
    use super::*;
    use crate::models::FitnessGoal;
    use crate::models::{Crossover, Mutagen, Strategy};

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let mutagen = Mutagen::constant(0.5, 0.1)?;
        let crossover = Crossover::uniform(0.5)?;
        let goal = FitnessGoal::maximize(0.9)?;

        let request = Request::new(
            "test",
            1,
            goal,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            mutagen,
            crossover,
        )?;
        let request_id = request.id;

        new_request(&pool, request).await?;
        let selected = get_request(&pool, &request_id).await?;

        assert_eq!(request_id, selected.id);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let mutagen = Mutagen::constant(0.5, 0.1)?;
        let crossover = Crossover::uniform(0.5)?;
        let goal = FitnessGoal::maximize(0.9)?;

        let request = Request::new(
            "test",
            1,
            goal,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            mutagen,
            crossover,
        )?;
        let request_id = request.id;

        let selected = get_request(&pool, &request_id).await;

        assert!(selected.is_err());
        Ok(())
    }
}
