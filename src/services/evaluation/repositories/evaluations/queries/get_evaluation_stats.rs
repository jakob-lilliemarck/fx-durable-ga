use super::super::models::EvaluationPopulation;
use super::super::Error as RepositoryError;
use chrono::{DateTime, Utc};
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Default)]
pub struct GetEvaluationStatsFilter {
    request_id: Option<Uuid>,
    genotype_id: Option<Uuid>,
    generated_since: Option<DateTime<Utc>>,
    generated_until: Option<DateTime<Utc>>,
    evaluated_by: Option<Uuid>,
    cursor: Option<Uuid>,
}

impl GetEvaluationStatsFilter {
    pub fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_id = Some(request_id);
        self
    }

    pub fn with_genotype_id(mut self, genotype_id: Uuid) -> Self {
        self.genotype_id = Some(genotype_id);
        self
    }

    pub fn with_generated_since(mut self, since: DateTime<Utc>) -> Self {
        self.generated_since = Some(since);
        self
    }

    pub fn with_generated_until(mut self, until: DateTime<Utc>) -> Self {
        self.generated_until = Some(until);
        self
    }

    pub fn with_evaluated_by(mut self, evaluated_by: Uuid) -> Self {
        self.evaluated_by = Some(evaluated_by);
        self
    }

    pub fn with_cursor(mut self, cursor: Uuid) -> Self {
        self.cursor = Some(cursor);
        self
    }
}

#[instrument(level = "debug", skip(tx), fields(filter = ?filter, limit = limit))]
pub async fn get_evaluation_stats<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &GetEvaluationStatsFilter,
    limit: i64,
) -> Result<EvaluationPopulation, RepositoryError> {
    let stats = sqlx::query!(
        r#"
            WITH filtered AS (
                SELECT fitness
                FROM evaluation.evaluations e
                WHERE ($1::uuid IS NULL OR e.request_id = $1)
                  AND ($2::uuid IS NULL OR e.genotype_id = $2)
                  AND ($3::timestamptz IS NULL OR e.generated_at >= $3)
                  AND ($4::timestamptz IS NULL OR e.generated_at <= $4)
                  AND ($5::uuid IS NULL OR e.evaluated_by = $5)
                  AND ($6::uuid IS NULL OR (e.generated_at, e.id) < (
                      SELECT cp.generated_at, cp.id
                      FROM evaluation.evaluations cp
                      WHERE cp.id = $6
                  ))
                ORDER BY e.generated_at DESC, e.id DESC
                LIMIT $7
            )
            SELECT
                COUNT(*) AS "count!:i64",
                MIN(fitness) AS "min?",
                MAX(fitness) AS "max?"
            FROM filtered
        "#,
        filter.request_id,
        filter.genotype_id,
        filter.generated_since,
        filter.generated_until,
        filter.evaluated_by,
        filter.cursor,
        limit,
    )
    .fetch_one(tx)
    .await?;

    Ok(EvaluationPopulation {
        evaluated_genotypes: stats.count,
        min_fitness: stats.min,
        max_fitness: stats.max,
    })
}

#[cfg(test)]
mod tests_get_evaluation_stats {
    use super::{GetEvaluationStatsFilter, get_evaluation_stats};
    use crate::services::evaluation::repositories::evaluations::Evaluation;
    use crate::services::evaluation::repositories::evaluations::queries::store_evaluations;
    use crate::services::evaluation::repositories::evaluations::queries::search_evaluations;
    use crate::services::evaluation::repositories::evaluations::SearchEvaluationsFilter;
    use crate::repositories::genotypes::Genotype;
    use crate::repositories::genotypes::store_genotypes;
    use crate::services::optimization::store_request;
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
    use chrono::{Duration, Utc};
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: &PgPool) -> anyhow::Result<(Uuid, Uuid, Vec<Uuid>)> {
        let request_a = Request::new(
            "test_a",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request_a = store_request(pool, request_a).await?;

        let request_b = Request::new(
            "test_b",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request_b = store_request(pool, request_b).await?;

        let make_gen = |req_id, data| {
            Genotype::new(
                "test", 1, serde_json::json!(data), Some(req_id), Some(1), None, None,
            )
        };

        let a_genotypes = store_genotypes(
            pool,
            &[
                make_gen(request_a.id, [1])?,
                make_gen(request_a.id, [2])?,
                make_gen(request_a.id, [3])?,
            ],
        )
        .await?;

        let b_genotypes = store_genotypes(
            pool,
            &[make_gen(request_b.id, [4])?, make_gen(request_b.id, [5])?],
        )
        .await?;

        let now = Utc::now();
        let host_id = Uuid::now_v7();

        let a_evals: Vec<Evaluation> = a_genotypes
            .iter()
            .enumerate()
            .map(|(i, g)| {
                Evaluation::new(
                    g.id(),
                    0.1 * (i as f64 + 1.0),
                    Some(now),
                    Some(now),
                    Some(host_id),
                )
                .with_request_id(request_a.id)
                .with_generated_at(now - Duration::seconds((a_genotypes.len() - i) as i64))
            })
            .collect();
        store_evaluations(pool, &a_evals).await?;

        let b_evals: Vec<Evaluation> = b_genotypes
            .iter()
            .enumerate()
            .map(|(i, g)| {
                Evaluation::new(
                    g.id(),
                    0.5 + 0.1 * (i as f64),
                    Some(now),
                    Some(now),
                    Some(host_id),
                )
                .with_request_id(request_b.id)
                .with_generated_at(now - Duration::seconds((b_genotypes.len() - i) as i64))
            })
            .collect();
        store_evaluations(pool, &b_evals).await?;

        let mut all_ids = a_genotypes
            .into_iter()
            .chain(b_genotypes)
            .map(|g| g.id())
            .collect::<Vec<_>>();
        all_ids.sort();

        Ok((request_a.id, request_b.id, all_ids))
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_stats_for_request(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (req_a, req_b, _) = seed(&pool).await?;

        let stats = get_evaluation_stats(
            &pool,
            &GetEvaluationStatsFilter::default().with_request_id(req_a),
            i64::MAX,
        )
        .await?;
        assert_eq!(stats.evaluated_genotypes, 3);
        assert!((stats.min_fitness.unwrap() - 0.1).abs() < 1e-12);
        assert!((stats.max_fitness.unwrap() - 0.3).abs() < 1e-12);

        let stats = get_evaluation_stats(
            &pool,
            &GetEvaluationStatsFilter::default().with_request_id(req_b),
            i64::MAX,
        )
        .await?;
        assert_eq!(stats.evaluated_genotypes, 2);
        assert!((stats.min_fitness.unwrap() - 0.5).abs() < 1e-12);
        assert!((stats.max_fitness.unwrap() - 0.6).abs() < 1e-12);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_zero_counts_for_nonexistent_request(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        seed(&pool).await?;

        let stats = get_evaluation_stats(
            &pool,
            &GetEvaluationStatsFilter::default().with_request_id(Uuid::nil()),
            i64::MAX,
        )
        .await?;
        assert_eq!(stats.evaluated_genotypes, 0);
        assert!(stats.min_fitness.is_none());
        assert!(stats.max_fitness.is_none());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_respects_limit(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (req_a, _, _) = seed(&pool).await?;

        let stats = get_evaluation_stats(
            &pool,
            &GetEvaluationStatsFilter::default().with_request_id(req_a),
            2,
        )
        .await?;
        assert_eq!(stats.evaluated_genotypes, 2);
        assert!((stats.min_fitness.unwrap() - 0.2).abs() < 1e-12);
        assert!((stats.max_fitness.unwrap() - 0.3).abs() < 1e-12);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_evaluated_by(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (req_a, _, _) = seed(&pool).await?;

        let stats = get_evaluation_stats(
            &pool,
            &GetEvaluationStatsFilter::default()
                .with_request_id(req_a)
                .with_evaluated_by(Uuid::nil()),
            i64::MAX,
        )
        .await?;
        assert_eq!(stats.evaluated_genotypes, 0);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_paginates_with_cursor(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (req_a, _, _) = seed(&pool).await?;

        let evals = search_evaluations(
            &pool,
            &SearchEvaluationsFilter::default()
                .with_request_ids(vec![req_a])
                .with_order_fitness_asc(),
        )
        .await?;
        assert_eq!(evals.len(), 3);

        let cursor = evals[1].id();
        let stats = get_evaluation_stats(
            &pool,
            &GetEvaluationStatsFilter::default()
                .with_request_id(req_a)
                .with_cursor(cursor),
            i64::MAX,
        )
        .await?;
        assert_eq!(stats.evaluated_genotypes, 1);
        assert!((stats.min_fitness.unwrap() - 0.1).abs() < 1e-12);

        Ok(())
    }
}
