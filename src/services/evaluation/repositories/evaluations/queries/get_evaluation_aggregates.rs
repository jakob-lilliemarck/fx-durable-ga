use super::super::Error;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[derive(Default, Debug)]
pub struct GetEvaluationAggregatesFilter {
    group_id: Option<Uuid>,
    genotype_id: Option<Uuid>,
}

impl GetEvaluationAggregatesFilter {
    pub fn with_group_id(mut self, group_id: Uuid) -> Self {
        self.group_id = Some(group_id);
        self
    }

    pub fn with_genotype_id(mut self, genotype_id: Uuid) -> Self {
        self.genotype_id = Some(genotype_id);
        self
    }
}

#[derive(Debug, Clone)]
pub struct EvaluationAggregates {
    pub count: i64,
    pub avg_fitness: Option<f64>,
    pub stddev_fitness: Option<f64>,
    pub variance_fitness: Option<f64>,
}

#[instrument(level = "debug", skip(tx), fields(filter = ?filter))]
pub async fn get_evaluation_aggregates<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &GetEvaluationAggregatesFilter,
) -> Result<EvaluationAggregates, Error> {
    let result = sqlx::query_as!(
        EvaluationAggregates,
        r#"
            SELECT
                COUNT(*)::bigint AS "count!",
                AVG(fitness) AS "avg_fitness",
                STDDEV_POP(fitness) AS "stddev_fitness",
                VAR_POP(fitness) AS "variance_fitness"
            FROM evaluation.evaluations
            WHERE ($1::uuid IS NULL OR genotype_id = $1)
              AND ($2::uuid IS NULL OR group_id = $2)
        "#,
        filter.genotype_id,
        filter.group_id,
    )
    .fetch_one(tx)
    .await?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migrations;
    use crate::repositories::genotypes::{Genotype, store_genotypes};
    use crate::services::evaluation::repositories::evaluations::{Evaluation, WriteTx};
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector, store_request};
    use chrono::Utc;
    use sqlx::PgPool;

    async fn create_request(pool: &PgPool) -> Uuid {
        let request = Request::new(
            "test",
            FitnessGoal::maximize(1.0).unwrap(),
            Selector::tournament(3),
            Schedule::generational(10, 2),
        );
        let id = request.id;
        store_request(pool, request).await.unwrap();
        id
    }

    async fn seed_evaluation(pool: &PgPool, genotype_id: Uuid, fitness: f64) {
        let mut tx = pool.begin().await.unwrap();
        let evaluation = Evaluation::new(
            genotype_id,
            Uuid::nil(),
            "test".to_string(),
            fitness,
            Some(Utc::now()),
            Some(Utc::now()),
            None,
        );
        let mut wr = WriteTx::new(&mut tx);
        wr.store_evaluations(&[evaluation]).await.unwrap();
        tx.commit().await.unwrap();
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_zero_count_when_no_match(pool: PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;
        let result = get_evaluation_aggregates(
            &pool,
            &GetEvaluationAggregatesFilter::default().with_genotype_id(Uuid::now_v7()),
        )
        .await?;
        assert_eq!(result.count, 0);
        assert!(result.avg_fitness.is_none());
        assert!(result.stddev_fitness.is_none());
        assert!(result.variance_fitness.is_none());
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_aggregates_for_genotype(pool: PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;
        let request_id = create_request(&pool).await;
        let genotype = Genotype::new("test", serde_json::json!([1]), request_id, None, None, None)?;
        let stored = store_genotypes(&pool, &[genotype]).await?;
        let genotype_id = stored[0].id();

        seed_evaluation(&pool, genotype_id, 0.5).await;
        seed_evaluation(&pool, genotype_id, 1.0).await;

        let result = get_evaluation_aggregates(
            &pool,
            &GetEvaluationAggregatesFilter::default().with_genotype_id(genotype_id),
        )
        .await?;
        assert_eq!(result.count, 2);
        assert!((result.avg_fitness.unwrap() - 0.75).abs() < 1e-10);
        Ok(())
    }
}
