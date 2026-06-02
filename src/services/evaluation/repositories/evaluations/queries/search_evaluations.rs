use super::super::Error as RepositoryError;
use super::super::Evaluation;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Default)]
pub struct SearchEvaluationsFilter {
    request_ids: Option<Vec<Uuid>>,
    genotype_ids: Option<Vec<Uuid>>,
    order_fitness: Option<&'static str>,
    order_completed_at: Option<&'static str>,
    limit: Option<i64>,
}

impl SearchEvaluationsFilter {
    pub fn with_request_ids(mut self, request_ids: Vec<Uuid>) -> Self {
        self.request_ids = Some(request_ids);
        self
    }

    pub fn with_genotype_ids(mut self, genotype_ids: Vec<Uuid>) -> Self {
        self.genotype_ids = Some(genotype_ids);
        self
    }

    pub fn with_order_fitness_asc(mut self) -> Self {
        self.order_fitness = Some("ASC");
        self
    }

    pub fn with_order_fitness_desc(mut self) -> Self {
        self.order_fitness = Some("DESC");
        self
    }

    pub fn with_order_completed_at_asc(mut self) -> Self {
        self.order_completed_at = Some("ASC");
        self
    }

    pub fn with_order_completed_at_desc(mut self) -> Self {
        self.order_completed_at = Some("DESC");
        self
    }

    pub fn with_limit(mut self, limit: i64) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Searches evaluations with optional filtering, ordering, and limits.
/// Returns full Evaluation records.
#[instrument(level = "debug", skip(tx))]
pub async fn search_evaluations<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &SearchEvaluationsFilter,
) -> Result<Vec<Evaluation>, RepositoryError> {
    let mut order_by = String::new();
    if let Some(dir) = filter.order_fitness {
        order_by = format!("ORDER BY fitness {dir}");
    } else if let Some(dir) = filter.order_completed_at {
        order_by = format!("ORDER BY completed_at {dir}");
    }

    let limit_clause = filter
        .limit
        .map(|l| format!("LIMIT {l}"))
        .unwrap_or_default();

    let sql = format!(
        r#"
            SELECT
                id, genotype_id, fitness, started_at, completed_at,
                evaluated_by, request_id, generated_at
            FROM evaluation.evaluations
            WHERE ($1::uuid[] IS NULL OR request_id = ANY($1))
              AND ($2::uuid[] IS NULL OR genotype_id = ANY($2))
            {order_by}
            {limit_clause}
        "#
    );

    let rows = sqlx::query_as::<_, Evaluation>(&sql)
        .bind(filter.request_ids.as_deref())
        .bind(filter.genotype_ids.as_deref())
        .fetch_all(tx)
        .await?;

    Ok(rows)
}

#[cfg(test)]
mod tests_search_evaluations {
    use super::{SearchEvaluationsFilter, search_evaluations};
    use crate::repositories::genotypes::Genotype;
    use crate::repositories::genotypes::store_genotypes;
    use crate::services::evaluation::repositories::evaluations::Evaluation;
    use crate::services::evaluation::repositories::evaluations::queries::store_evaluations;
    use crate::services::optimization::store_request;
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
    use chrono::Utc;
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: &PgPool) -> anyhow::Result<(Uuid, Uuid, Vec<Uuid>)> {
        let new_request = |name: &str| -> anyhow::Result<Request> {
            Ok(Request::new(
                name,
                FitnessGoal::maximize(0.9)?,
                Selector::tournament(10),
                Schedule::generational(100, 10),
            ))
        };

        let req_a = store_request(pool, new_request("test_a")?).await?;
        let req_b = store_request(pool, new_request("test_b")?).await?;

        let make_genotype = |req_id: Uuid, data: serde_json::Value| {
            Genotype::new("test", data, Some(req_id), Some(1), None, None)
        };

        let a_genotypes = vec![
            make_genotype(req_a.id, serde_json::json!([1]))?,
            make_genotype(req_a.id, serde_json::json!([2]))?,
            make_genotype(req_a.id, serde_json::json!([3]))?,
        ];
        let a_genotypes = store_genotypes(pool, &a_genotypes).await?;

        let b_genotypes = vec![
            make_genotype(req_b.id, serde_json::json!([4]))?,
            make_genotype(req_b.id, serde_json::json!([5]))?,
            make_genotype(req_b.id, serde_json::json!([6]))?,
        ];
        let b_genotypes = store_genotypes(pool, &b_genotypes).await?;

        let a_fitness = [0.1, 0.2, 0.3];
        let b_fitness = [0.4, 0.5, 0.6];

        let a_evals: Vec<Evaluation> = a_genotypes
            .iter()
            .zip(a_fitness.iter())
            .map(|(g, f)| {
                Evaluation::new(g.id(), *f, Some(Utc::now()), Some(Utc::now()), None)
                    .with_request_id(req_a.id)
                    .with_generated_at(g.generated_at())
            })
            .collect();

        let b_evals: Vec<Evaluation> = b_genotypes
            .iter()
            .zip(b_fitness.iter())
            .map(|(g, f)| {
                Evaluation::new(g.id(), *f, Some(Utc::now()), Some(Utc::now()), None)
                    .with_request_id(req_b.id)
                    .with_generated_at(g.generated_at())
            })
            .collect();

        let mut all_evals = a_evals;
        all_evals.extend(b_evals);
        store_evaluations(pool, &all_evals).await?;

        let mut all_ids = a_genotypes
            .into_iter()
            .chain(b_genotypes)
            .map(|g| g.id())
            .collect::<Vec<_>>();

        all_ids.sort();
        Ok((req_a.id, req_b.id, all_ids))
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_request_id(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (req_a, _, _) = seed(&pool).await?;

        let results = search_evaluations(
            &pool,
            &SearchEvaluationsFilter::default().with_request_ids(vec![req_a]),
        )
        .await?;

        assert_eq!(results.len(), 3);
        for eval in &results {
            assert_eq!(eval.request_id(), &Some(req_a));
        }
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_genotype_ids(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (_, _, all_ids) = seed(&pool).await?;

        let target_ids = vec![all_ids[0], all_ids[2]];
        let results = search_evaluations(
            &pool,
            &SearchEvaluationsFilter::default().with_genotype_ids(target_ids.clone()),
        )
        .await?;

        assert_eq!(results.len(), 2);
        let mut returned: Vec<Uuid> = results.into_iter().map(|e| *e.genotype_id()).collect();
        returned.sort();
        assert_eq!(returned, target_ids);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_orders_by_fitness_asc(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (req_a, _, _) = seed(&pool).await?;

        let results = search_evaluations(
            &pool,
            &SearchEvaluationsFilter::default()
                .with_request_ids(vec![req_a])
                .with_order_fitness_asc(),
        )
        .await?;

        assert_eq!(results.len(), 3);
        assert!((results[0].fitness() - 0.1).abs() < 1e-12);
        assert!((results[1].fitness() - 0.2).abs() < 1e-12);
        assert!((results[2].fitness() - 0.3).abs() < 1e-12);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_orders_by_fitness_desc(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (req_a, _, _) = seed(&pool).await?;

        let results = search_evaluations(
            &pool,
            &SearchEvaluationsFilter::default()
                .with_request_ids(vec![req_a])
                .with_order_fitness_desc(),
        )
        .await?;

        assert_eq!(results.len(), 3);
        assert!((results[0].fitness() - 0.3).abs() < 1e-12);
        assert!((results[1].fitness() - 0.2).abs() < 1e-12);
        assert!((results[2].fitness() - 0.1).abs() < 1e-12);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_limits_results(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        seed(&pool).await?;

        let results =
            search_evaluations(&pool, &SearchEvaluationsFilter::default().with_limit(2)).await?;

        assert_eq!(results.len(), 2);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_when_no_match(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let results = search_evaluations(
            &pool,
            &SearchEvaluationsFilter::default().with_request_ids(vec![Uuid::nil()]),
        )
        .await?;

        assert!(results.is_empty());
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_all_without_filters(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        seed(&pool).await?;

        let results = search_evaluations(&pool, &SearchEvaluationsFilter::default()).await?;

        assert_eq!(results.len(), 6);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_combines_filter_and_order_and_limit(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (req_a, _, _) = seed(&pool).await?;

        let results = search_evaluations(
            &pool,
            &SearchEvaluationsFilter::default()
                .with_request_ids(vec![req_a])
                .with_order_fitness_desc()
                .with_limit(2),
        )
        .await?;

        assert_eq!(results.len(), 2);
        assert!((results[0].fitness() - 0.3).abs() < 1e-12);
        assert!((results[1].fitness() - 0.2).abs() < 1e-12);
        Ok(())
    }
}
