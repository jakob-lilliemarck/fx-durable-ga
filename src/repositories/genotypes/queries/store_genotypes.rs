use super::super::Error as RepositoryError;
use crate::repositories::genotypes::Genotype;
use sqlx::PgExecutor;
use tracing::instrument;

/// Inserts multiple genotypes into the database in a single transaction.
/// Returns the inserted genotypes with database-generated fields.
#[instrument(level = "debug", skip(tx, genotypes))]
pub(crate) async fn store_genotypes<'tx, 'a, I, E>(
    tx: E,
    genotypes: I,
) -> Result<Vec<Genotype>, RepositoryError>
where
    I: IntoIterator<Item = &'a Genotype>,
    E: PgExecutor<'tx>,
{
    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.genotypes (
            id,
            generated_at,
            type_name,
            type_hash,
            genome,
            genome_hash,
            request_id,
            generation_id,
            parent_a,
            parent_b
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
            .push_bind(g.id())
            .push(", ")
            .push_bind(g.generated_at())
            .push(", ")
            .push_bind(g.type_name().to_string())
            .push(", ")
            .push_bind(g.type_hash())
            .push(", ")
            .push_bind(g.genome())
            .push(", ")
            .push_bind(g.genome_hash())
            .push(", ")
            .push_bind(g.request_id())
            .push(", ")
            .push_bind(g.generation_id())
            .push(", ")
            .push_bind(g.parent_a)
            .push(", ")
            .push_bind(g.parent_b)
            .push(")");
    }
    query_builder.push(
        " RETURNING id, generated_at, type_name, type_hash, genome, genome_hash, request_id, generation_id, parent_a, parent_b",
    );

    let genotypes = query_builder
        .build_query_as::<Genotype>()
        .fetch_all(tx)
        .await?;

    Ok(genotypes)
}

#[cfg(test)]
mod tests_store_genotypes {
    use super::*;
    use crate::services::optimization::store_request;
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
    use chrono::SubsecRound;

    #[sqlx::test(migrations = false)]
    async fn it_inserts_a_new_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request_id = request.id;
        store_request(&pool, request).await?;

        let genotypes = vec![Genotype::new(
            "test",
            1,
            serde_json::json!([1, 2, 3]),
            Some(request_id),
            Some(1),
            None,
            None,
        )?];
        let genotypes_clone = genotypes.clone();

        let inserted = store_genotypes(&pool, &genotypes).await?;

        assert_eq!(genotypes_clone[0].id(), inserted[0].id());
        assert_eq!(
            genotypes_clone[0].generated_at().trunc_subsecs(6),
            inserted[0].generated_at()
        );
        assert_eq!(genotypes_clone[0].type_name(), inserted[0].type_name());
        assert_eq!(genotypes_clone[0].type_hash(), inserted[0].type_hash());
        assert_eq!(genotypes_clone[0].genome(), inserted[0].genome());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request_id = request.id;
        store_request(&pool, request).await?;

        let genotype = Genotype::new(
            "test",
            1,
            serde_json::json!([1, 2, 3]),
            Some(request_id),
            Some(1),
            None,
            None,
        )?;
        let genotype_clone = genotype.clone();

        store_genotypes(&pool, &[genotype]).await?;
        let result = store_genotypes(&pool, &[genotype_clone]).await;

        assert!(result.is_err());
        Ok(())
    }
}
