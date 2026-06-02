use super::super::Error as RepositoryError;
use crate::repositories::genotypes::Genotype;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

/// Get the descendants of a genotype.
/// Limits recursive search to the specified `degree`.
#[instrument(level = "debug", skip(tx), fields(genotype_id = %genotype_id, degree=?degree))]
pub async fn get_descendants<'tx, E: PgExecutor<'tx>>(
    tx: E,
    genotype_id: &Uuid,
    degree: i32,
) -> Result<Vec<Genotype>, RepositoryError> {
    let genotypes = sqlx::query_as::<_, Genotype>(
        r#"
            WITH RECURSIVE descendant_tree AS (
                SELECT
                    g.id, g.generated_at, g.type_name,
                    g.genome, g.genome_hash, g.request_id, g.generation_id,
                    g.parent_a, g.parent_b,
                    0 AS degree
                FROM fx_durable_ga.genotypes g
                WHERE g.id = $1
                UNION
                SELECT
                    child.id, child.generated_at, child.type_name,
                    child.genome, child.genome_hash, child.request_id, child.generation_id,
                    child.parent_a, child.parent_b,
                    parent.degree + 1
                FROM descendant_tree parent
                JOIN fx_durable_ga.genotypes child
                ON child.parent_a = parent.id
                OR child.parent_b = parent.id
                WHERE parent.degree + 1 <= $2::INTEGER
            )
            SELECT
                id, generated_at, type_name,
                genome, genome_hash, request_id, generation_id,
                parent_a, parent_b
            FROM descendant_tree
            ORDER BY degree, id ASC;
        "#,
    )
    .bind(genotype_id)
    .bind(degree)
    .fetch_all(tx)
    .await?;

    Ok(genotypes)
}

#[cfg(test)]
mod tests_get_descendants {
    use super::super::test_tools::seed_lineage;
    use super::get_descendants;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_gets_each_descendant(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = seed_lineage(&pool).await;
        let descendants = get_descendants(&pool, &lineage[0], 5).await?;
        let ids: Vec<Uuid> = descendants.iter().map(|g| g.id()).collect();

        assert_eq!(ids, vec![lineage[0], lineage[1], lineage[2]]);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_is_bounded_by_degree(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = seed_lineage(&pool).await;
        let descendants = get_descendants(&pool, &lineage[0], 1).await?;
        let ids: Vec<Uuid> = descendants.iter().map(|g| g.id()).collect();

        assert_eq!(ids, vec![lineage[0], lineage[1]]);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_no_descendants_when_there_are_none(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = seed_lineage(&pool).await;
        let descendants = get_descendants(&pool, &lineage[2], 5).await?;
        let ids: Vec<Uuid> = descendants.iter().map(|g| g.id()).collect();

        assert_eq!(ids, vec![lineage[2]]);
        Ok(())
    }
}
