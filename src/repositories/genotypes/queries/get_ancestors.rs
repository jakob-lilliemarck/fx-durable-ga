use super::super::Error as RepositoryError;
use crate::repositories::genotypes::Genotype;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

/// Get the ancestors of a genotype.
/// Limits recursive search to the specified `degree`.
#[instrument(level = "debug", skip(tx), fields(genotype_id = %genotype_id, degree=?degree))]
pub async fn get_ancestors<'tx, E: PgExecutor<'tx>>(
    tx: E,
    genotype_id: &Uuid,
    degree: i32,
) -> Result<Vec<Genotype>, RepositoryError> {
    let genotypes = sqlx::query_as::<_, Genotype>(
        r#"
            WITH RECURSIVE ancestor_tree AS (
                SELECT
                    g.id, g.generated_at, g.type_name, g.type_hash,
                    g.genome, g.genome_hash, g.request_id, g.generation_id,
                    g.parent_a, g.parent_b,
                    0 AS degree
                FROM fx_durable_ga.genotypes g
                WHERE g.id = $1
                UNION
                SELECT
                    parent.id, parent.generated_at, parent.type_name, parent.type_hash,
                    parent.genome, parent.genome_hash, parent.request_id, parent.generation_id,
                    parent.parent_a, parent.parent_b,
                    child.degree + 1
                FROM ancestor_tree child
                JOIN fx_durable_ga.genotypes parent
                ON parent.id = child.parent_a
                OR parent.id = child.parent_b
                WHERE child.degree + 1 <= $2::INTEGER
            )
            SELECT
                id, generated_at, type_name, type_hash,
                genome, genome_hash, request_id, generation_id,
                parent_a, parent_b
            FROM ancestor_tree
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
mod tests_get_ancestors {
    use super::super::test_tools::seed_lineage;
    use super::get_ancestors;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_gets_each_ancestor(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = seed_lineage(&pool).await;
        let ancestors = get_ancestors(&pool, &lineage[2], 5).await?;
        let ids: Vec<Uuid> = ancestors.iter().map(|g| g.id()).collect();

        assert_eq!(ids, vec![lineage[2], lineage[1], lineage[0]]);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_is_bounded_by_degree(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = seed_lineage(&pool).await;
        let ancestors = get_ancestors(&pool, &lineage[2], 1).await?;
        let ids: Vec<Uuid> = ancestors.iter().map(|g| g.id()).collect();

        assert_eq!(ids, vec![lineage[2], lineage[1]]);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_no_ancestors_when_there_are_none(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = seed_lineage(&pool).await;
        let ancestors = get_ancestors(&pool, &lineage[0], 5).await?;
        let ids: Vec<Uuid> = ancestors.iter().map(|g| g.id()).collect();

        assert_eq!(ids, vec![lineage[0]]);
        Ok(())
    }
}
