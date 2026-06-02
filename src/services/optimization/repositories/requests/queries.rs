use super::Error;
use super::models::DbRequest;
use crate::services::optimization::Request;
use chrono::{DateTime, Utc};
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

/// Store a optimization request
#[instrument(level = "debug", skip(tx), fields(
    request_id = %request.id,
    type_name = %request.type_name,
    goal = ?request.goal
))]
pub(crate) async fn store_request<'tx, E: PgExecutor<'tx>>(
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
                goal,
                schedule,
                selector,
                account_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING
                id,
                requested_at,
                type_name,
                goal,
                schedule,
                selector,
                account_id;
            "#,
        db_request.id,
        db_request.requested_at,
        db_request.type_name,
        db_request.goal,
        db_request.schedule,
        db_request.selector,
        db_request.account_id
    )
    .fetch_one(tx)
    .await?;

    let request: Request = db_request.try_into()?;
    Ok(request)
}

#[cfg(test)]
mod tests_store_request {
    use super::*;
    use crate::services::optimization::{FitnessGoal, Schedule, Selector};
    use chrono::SubsecRound;

    #[sqlx::test(migrations = false)]
    async fn it_stores_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let goal = FitnessGoal::maximize(0.9)?;

        let request = Request::new(
            "test",
            goal,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request_clone = request.clone();

        let inserted = store_request(&pool, request).await?;

        assert_eq!(request_clone.id, inserted.id);
        assert_eq!(
            request_clone.requested_at.trunc_subsecs(6),
            inserted.requested_at
        );
        assert_eq!(request_clone.type_name, inserted.type_name);
        assert_eq!(request_clone.goal, inserted.goal);
        assert_eq!(request_clone.schedule, inserted.schedule);
        assert_eq!(request_clone.selector, inserted.selector);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let goal = FitnessGoal::maximize(0.9)?;

        let request = Request::new(
            "test",
            goal,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request_clone = request.clone();

        store_request(&pool, request).await?;
        let inserted = store_request(&pool, request_clone).await;

        assert!(inserted.is_err());

        Ok(())
    }
}

#[derive(Debug)]
pub struct SearchRequestsFilter {
    cursor: Option<Uuid>,
    search: Option<String>,
    since: Option<DateTime<Utc>>,
    until: Option<DateTime<Utc>>,
    account_id: Option<Uuid>,
}

impl std::default::Default for SearchRequestsFilter {
    fn default() -> Self {
        Self {
            cursor: None,
            search: None,
            since: None,
            until: None,
            account_id: None,
        }
    }
}

impl SearchRequestsFilter {
    pub fn with_cursor(mut self, cursor: &Uuid) -> Self {
        self.cursor = Some(*cursor);
        self
    }

    pub fn with_search(mut self, search: String) -> Self {
        self.search = Some(search);
        self
    }

    pub fn with_since(mut self, since: DateTime<Utc>) -> Self {
        self.since = Some(since);
        self
    }

    pub fn with_until(mut self, until: DateTime<Utc>) -> Self {
        self.until = Some(until);
        self
    }

    pub fn with_account_id(mut self, account_id: Uuid) -> Self {
        self.account_id = Some(account_id);
        self
    }
}

#[instrument(level = "debug", skip(tx))]
pub async fn search_requests<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &SearchRequestsFilter,
    limit: i64,
) -> Result<Vec<Request>, Error> {
    let db_requests = sqlx::query_as!(
        DbRequest,
        r#"
        SELECT
            id,
            requested_at,
            type_name,
            goal,
            schedule,
            selector,
            account_id
        FROM fx_durable_ga.requests
        WHERE (
            $1::UUID IS NULL
            OR (requested_at, id) < (
                SELECT requested_at, id
                FROM fx_durable_ga.requests
                WHERE id = $1
            )
        )
        AND ($2::TEXT IS NULL OR type_name ILIKE $2 OR id::TEXT ILIKE $2)
        AND ($3::TIMESTAMPTZ IS NULL OR requested_at >= $3)
        AND ($4::TIMESTAMPTZ IS NULL OR requested_at <= $4)
        AND ($5::UUID IS NULL OR account_id = $5)
        ORDER BY requested_at DESC, id DESC
        LIMIT $6
        "#,
        filter.cursor.as_ref(),
        filter.search.as_ref().map(|q| format!("%{}%", q)),
        filter.since.as_ref(),
        filter.until.as_ref(),
        filter.account_id.as_ref(),
        limit,
    )
    .fetch_all(tx)
    .await?;

    db_requests.into_iter().map(|dbr| dbr.try_into()).collect()
}

#[cfg(test)]
mod tests_search_requests {
    use super::{SearchRequestsFilter, search_requests, store_request};
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
    use chrono::{Duration as ChronoDuration, SubsecRound, Utc};

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_since(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let seeded = seed(&pool).await?;

        let results = search_requests(
            &pool,
            &SearchRequestsFilter::default().with_since(seeded.timestamps[1]),
            10,
        )
        .await?;

        let actual = results.into_iter().map(|req| req.id).collect::<Vec<_>>();
        assert_eq!(
            vec![
                seeded.requests[3].id,
                seeded.requests[2].id,
                seeded.requests[1].id
            ],
            actual
        );
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_until(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let seeded = seed(&pool).await?;

        let results = search_requests(
            &pool,
            &SearchRequestsFilter::default().with_until(seeded.timestamps[1]),
            10,
        )
        .await?;

        let actual = results.into_iter().map(|req| req.id).collect::<Vec<_>>();
        assert_eq!(vec![seeded.requests[1].id, seeded.requests[0].id], actual);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_query_on_type_name(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let seeded = seed(&pool).await?;

        let results = search_requests(
            &pool,
            &SearchRequestsFilter::default().with_search("type_c".to_string()),
            10,
        )
        .await?;

        let actual = results.into_iter().map(|req| req.id).collect::<Vec<_>>();
        assert_eq!(vec![seeded.requests[2].id], actual);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_query_on_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let seeded = seed(&pool).await?;

        let id_query = seeded.requests[3].id.to_string();
        let results = search_requests(
            &pool,
            &SearchRequestsFilter::default().with_search(id_query),
            10,
        )
        .await?;

        let actual = results.into_iter().map(|req| req.id).collect::<Vec<_>>();
        assert_eq!(vec![seeded.requests[3].id], actual);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_paginates_by_cursor(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let seeded = seed(&pool).await?;

        let first_page = search_requests(&pool, &SearchRequestsFilter::default(), 2).await?;

        let first_ids = first_page.iter().map(|req| req.id).collect::<Vec<_>>();
        assert_eq!(
            vec![seeded.requests[3].id, seeded.requests[2].id],
            first_ids
        );

        let last_id = first_page
            .last()
            .expect("expected first page to have results")
            .id;

        let second_page = search_requests(
            &pool,
            &SearchRequestsFilter::default().with_cursor(&last_id),
            2,
        )
        .await?;

        let second_ids = second_page.iter().map(|req| req.id).collect::<Vec<_>>();
        assert_eq!(
            vec![seeded.requests[1].id, seeded.requests[0].id],
            second_ids
        );
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_vector_on_no_match(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        seed(&pool).await?;

        let results = search_requests(
            &pool,
            &SearchRequestsFilter::default().with_search("no_such_request".to_string()),
            10,
        )
        .await?;

        assert!(results.is_empty());

        Ok(())
    }

    struct SeedData {
        requests: Vec<Request>,
        timestamps: Vec<chrono::DateTime<Utc>>,
    }

    async fn seed(pool: &sqlx::PgPool) -> anyhow::Result<SeedData> {
        let base_time = Utc::now().trunc_subsecs(6);
        let timestamps = vec![
            base_time,
            base_time + ChronoDuration::seconds(1),
            base_time + ChronoDuration::seconds(2),
            base_time + ChronoDuration::seconds(3),
        ];

        let mut requests = Vec::with_capacity(4);
        for (i, &ts) in timestamps.iter().enumerate() {
            let type_name = format!("type_{}", (b'a' + i as u8) as char);
            let mut request = Request::new(
                &type_name,
                FitnessGoal::maximize(0.9)?,
                Selector::tournament(10),
                Schedule::generational(100, 10),
            );
            request.requested_at = ts;
            store_request(pool, request.clone()).await?;
            requests.push(request);
        }

        Ok(SeedData {
            requests,
            timestamps,
        })
    }
}

/// Retrieves a request by id
#[instrument(level = "debug", skip(tx), fields(request_id = %id))]
pub async fn get_request<'tx, E: PgExecutor<'tx>>(tx: E, id: &Uuid) -> Result<Request, Error> {
    let db_request = sqlx::query_as!(
        DbRequest,
        r#"
        SELECT
            id,
            requested_at,
            type_name,
            goal,
            schedule,
            selector,
            account_id
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
mod tests_get_request {
    use super::*;
    use crate::services::optimization::{FitnessGoal, Schedule, Selector};

    #[sqlx::test(migrations = false)]
    async fn it_gets_request(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let goal = FitnessGoal::maximize(0.9)?;

        let request = Request::new(
            "test",
            goal,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request_id = request.id;

        store_request(&pool, request).await?;
        let selected = get_request(&pool, &request_id).await?;

        assert_eq!(request_id, selected.id);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let not_found_id = Uuid::nil();
        let selected = get_request(&pool, &not_found_id).await;

        assert!(selected.is_err());
        Ok(())
    }
}
