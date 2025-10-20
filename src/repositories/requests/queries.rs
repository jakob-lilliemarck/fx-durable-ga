use super::Error;
use super::models::DbRequest;
use crate::models::{Conclusion, Request, RequestConclusion};
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

/// Creates a new request record in the database.
/// 
/// Returns the created request with database-assigned fields populated.
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
                schedule,
                selector,
                mutagen,
                crossover,
                distribution
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING
                id,
                requested_at,
                type_name,
                type_hash,
                goal,
                schedule,
                selector,
                mutagen,
                crossover,
                distribution;
            "#,
        db_request.id,
        db_request.requested_at,
        db_request.type_name,
        db_request.type_hash,
        db_request.goal,
        db_request.schedule,
        db_request.selector,
        db_request.mutagen,
        db_request.crossover,
        db_request.distribution
    )
    .fetch_one(tx)
    .await?;

    let request: Request = db_request.try_into()?;
    Ok(request)
}

#[cfg(test)]
mod new_request_tests {
    use super::*;
    use crate::models::{Crossover, Distribution, FitnessGoal, Mutagen, Schedule, Selector};
    use chrono::SubsecRound;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let mutagen = Mutagen::constant(0.5, 0.1)?;
        let crossover = Crossover::uniform(0.5)?;
        let goal = FitnessGoal::maximize(0.9)?;
        let distribution = Distribution::latin_hypercube(200);

        let request = Request::new(
            "test",
            1,
            goal,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            mutagen,
            crossover,
            distribution,
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
        assert_eq!(request_clone.schedule, inserted.schedule);
        assert_eq!(request_clone.selector, inserted.selector);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let mutagen = Mutagen::constant(0.5, 0.1)?;
        let crossover = Crossover::uniform(0.5)?;
        let goal = FitnessGoal::maximize(0.9)?;
        let distribution = Distribution::latin_hypercube(200);

        let request = Request::new(
            "test",
            1,
            goal,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            mutagen,
            crossover,
            distribution,
        )?;
        let request_clone = request.clone();

        new_request(&pool, request).await?;
        let inserted = new_request(&pool, request_clone).await;

        assert!(inserted.is_err());

        Ok(())
    }
}

/// Retrieves a request record from the database by ID.
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
            schedule,
            selector,
            mutagen,
            crossover,
            distribution
        FROM fx_durable_ga.requests
        WHERE id = $1;
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
    use crate::models::{Crossover, Distribution, FitnessGoal, Mutagen, Schedule, Selector};

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let mutagen = Mutagen::constant(0.5, 0.1)?;
        let crossover = Crossover::uniform(0.5)?;
        let goal = FitnessGoal::maximize(0.9)?;
        let distribution = Distribution::latin_hypercube(200);

        let request = Request::new(
            "test",
            1,
            goal,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            mutagen,
            crossover,
            distribution,
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
        let distribution = Distribution::latin_hypercube(200);

        let request = Request::new(
            "test",
            1,
            goal,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            mutagen,
            crossover,
            distribution,
        )?;
        let request_id = request.id;

        let selected = get_request(&pool, &request_id).await;

        assert!(selected.is_err());
        Ok(())
    }
}

/// Creates a request conclusion record in the database.
/// 
/// Returns the created conclusion with database-assigned fields populated.
#[instrument(level = "debug", skip(tx), fields(request_id = %request_conclusion.request_id, concluded_at = %request_conclusion.concluded_at, concluded_with=?request_conclusion))]
pub(crate) async fn new_request_conclusion<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_conclusion: &RequestConclusion,
) -> Result<RequestConclusion, Error> {
    let request_conclusion = sqlx::query_as!(
        RequestConclusion,
        r#"
            INSERT INTO fx_durable_ga.request_conclusions (
                request_id,
                concluded_at,
                concluded_with
            ) VALUES ($1, $2, $3)
            RETURNING
                request_id,
                concluded_at,
                concluded_with as "concluded_with: Conclusion";
        "#,
        request_conclusion.request_id,
        request_conclusion.concluded_at,
        request_conclusion.concluded_with as Conclusion
    )
    .fetch_one(tx)
    .await?;

    Ok(request_conclusion)
}

#[cfg(test)]
mod new_request_conclusion_conclusion_tests {
    use crate::models::Crossover;
    use crate::models::Distribution;
    use crate::models::FitnessGoal;
    use crate::models::Mutagen;
    use crate::models::Request;
    use crate::models::Schedule;
    use crate::models::Selector;
    use crate::repositories::requests::queries::new_request;
    use crate::{
        models::{Conclusion, RequestConclusion},
        repositories::requests::queries::new_request_conclusion,
    };
    use chrono::SubsecRound;
    use chrono::Utc;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_completed_conclusion(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let request_conclusion = RequestConclusion {
            request_id,
            concluded_at: Utc::now(),
            concluded_with: Conclusion::Completed,
        };

        let actual = new_request_conclusion(&pool, &request_conclusion).await?;

        assert_eq!(request_conclusion.request_id, actual.request_id);
        assert_eq!(
            request_conclusion.concluded_at.trunc_subsecs(6),
            actual.concluded_at
        );
        assert_eq!(request_conclusion.concluded_with, actual.concluded_with);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_terminated_conclusion(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let request_conclusion = RequestConclusion {
            request_id,
            concluded_at: Utc::now(),
            concluded_with: Conclusion::Terminated,
        };

        let actual = new_request_conclusion(&pool, &request_conclusion).await?;

        assert_eq!(request_conclusion.request_id, actual.request_id);
        assert_eq!(
            request_conclusion.concluded_at.trunc_subsecs(6),
            actual.concluded_at
        );
        assert_eq!(request_conclusion.concluded_with, actual.concluded_with);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let request_conclusion = RequestConclusion {
            request_id,
            concluded_at: Utc::now(),
            concluded_with: Conclusion::Terminated,
        };

        let first = new_request_conclusion(&pool, &request_conclusion).await;
        let second = new_request_conclusion(&pool, &request_conclusion).await;

        assert!(first.is_ok());
        assert!(second.is_err());

        Ok(())
    }
}

/// Retrieves a request conclusion from the database by request ID.
/// 
/// Returns None if no conclusion exists for the given request.
#[instrument(level = "debug", skip(tx), fields(id = %id))]
pub(crate) async fn get_request_conclusion<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: &Uuid,
) -> Result<Option<RequestConclusion>, Error> {
    let request_conclusion = sqlx::query_as!(
        RequestConclusion,
        r#"
            SELECT
                request_id,
                concluded_at,
                concluded_with as "concluded_with: Conclusion"
            FROM fx_durable_ga.request_conclusions
            WHERE request_id = $1;
        "#,
        id
    )
    .fetch_optional(tx)
    .await?;

    Ok(request_conclusion)
}

#[cfg(test)]
mod get_request_conclusion_tests {
    use crate::models::Crossover;
    use crate::models::Distribution;
    use crate::models::FitnessGoal;
    use crate::models::Mutagen;
    use crate::models::Request;
    use crate::models::Schedule;
    use crate::models::Selector;
    use crate::repositories::requests::queries::get_request_conclusion;
    use crate::repositories::requests::queries::new_request;
    use crate::{
        models::{Conclusion, RequestConclusion},
        repositories::requests::queries::new_request_conclusion,
    };
    use chrono::SubsecRound;
    use chrono::Utc;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_a_request_conclusion(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let request_conclusion = RequestConclusion {
            request_id,
            concluded_at: Utc::now(),
            concluded_with: Conclusion::Terminated,
        };
        new_request_conclusion(&pool, &request_conclusion).await?;

        let actual = get_request_conclusion(&pool, &request_id)
            .await?
            .expect("expected be Some(RequestConclusion)");

        assert_eq!(request_conclusion.request_id, actual.request_id);
        assert_eq!(
            request_conclusion.concluded_at.trunc_subsecs(6),
            actual.concluded_at
        );
        assert_eq!(request_conclusion.concluded_with, actual.concluded_with);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_returns_none_when_there_is_none(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let actual = get_request_conclusion(&pool, &request_id).await?;
        assert!(actual.is_none());

        Ok(())
    }
}
