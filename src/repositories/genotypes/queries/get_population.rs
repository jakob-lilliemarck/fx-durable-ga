use super::super::Error as RepositoryError;
use super::super::GenotypePopulation;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

/// Gets genotype-level population statistics for an optimization request.
/// Evaluation-level stats (fitness) must be fetched from the evaluations repo.
#[instrument(level = "debug", skip(tx), fields(request_id = %request_id))]
pub async fn get_population<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: &Uuid,
) -> Result<GenotypePopulation, RepositoryError> {
    let population = sqlx::query_as::<_, GenotypePopulation>(
        r#"
            SELECT
                $1::uuid AS request_id,
                COUNT(*)::bigint AS total_genotypes,
                COALESCE(MAX(generation_id), 0)::int AS current_generation
            FROM fx_durable_ga.genotypes
            WHERE request_id = $1
        "#,
    )
    .bind(request_id)
    .fetch_one(tx)
    .await?;

    Ok(population)
}

#[cfg(test)]
mod tests_get_population {
    use super::get_population;
    use crate::repositories::genotypes::Genotype;
    use crate::repositories::genotypes::store_genotypes;
    use crate::services::optimization::store_request;
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: PgPool) -> anyhow::Result<Uuid> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request = store_request(&pool, request).await?;

        let genotypes = vec![
            Genotype::new(
                "test",
                1,
                serde_json::json!([1, 2, 3]),
                Some(request.id),
                Some(1),
                None,
                None,
            )?,
            Genotype::new(
                "test",
                1,
                serde_json::json!([4, 5, 6]),
                Some(request.id),
                Some(1),
                None,
                None,
            )?,
            Genotype::new(
                "test",
                1,
                serde_json::json!([7, 8, 9]),
                Some(request.id),
                Some(1),
                None,
                None,
            )?,
        ];
        store_genotypes(&pool, &genotypes).await?;
        Ok(request.id)
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_population(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let request_id = seed(pool.clone()).await?;

        let population = get_population(&pool, &request_id).await?;
        assert_eq!(population.total_genotypes(), 3);
        assert_eq!(population.current_generation(), 1);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_empty_populations(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let request_id = Uuid::nil();
        let population = get_population(&pool, &request_id).await?;

        assert_eq!(population.total_genotypes(), 0);
        assert_eq!(population.current_generation(), 0);

        Ok(())
    }
}
