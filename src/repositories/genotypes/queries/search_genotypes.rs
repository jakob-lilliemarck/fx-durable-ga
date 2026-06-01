use super::super::Error as RepositoryError;
use crate::repositories::genotypes::Genotype;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

/// Filter criteria for searching genotypes with various conditions.
#[derive(Debug)]
pub struct SearchGenotypesFilter {
    request_ids: Option<Vec<Uuid>>,
    generation_ids: Option<Vec<i32>>,
    genotype_ids: Option<Vec<Uuid>>,
    order_random: bool,
    cursor: Option<Uuid>,
    search: Option<String>,
}

impl Default for SearchGenotypesFilter {
    fn default() -> Self {
        SearchGenotypesFilter {
            request_ids: None,
            generation_ids: None,
            genotype_ids: None,
            order_random: false,
            cursor: None,
            search: None,
        }
    }
}

impl SearchGenotypesFilter {
    pub fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_ids
            .get_or_insert_with(Vec::new)
            .push(request_id);
        self
    }

    #[allow(dead_code)]
    pub fn with_generation_id(mut self, generation_id: i32) -> Self {
        self.generation_ids
            .get_or_insert_with(Vec::new)
            .push(generation_id);
        self
    }

    #[allow(dead_code)]
    pub fn with_genotype_id(mut self, genotype_id: Uuid) -> Self {
        self.genotype_ids
            .get_or_insert_with(Vec::new)
            .push(genotype_id);
        self
    }

    pub fn with_genotype_ids(mut self, genotype_ids: Vec<Uuid>) -> Self {
        self.genotype_ids = Some(genotype_ids);
        self
    }

    pub fn with_order_random(mut self) -> Self {
        self.order_random = true;
        self
    }

    pub fn with_cursor(mut self, cursor: Uuid) -> Self {
        self.cursor = Some(cursor);
        self
    }

    pub fn with_search(mut self, search: String) -> Self {
        self.search = Some(search);
        self
    }
}

/// Searches genotypes with optional filtering, ordering, and limits.
#[instrument(level = "debug", skip(tx), fields(filter = ?filter))]
pub async fn search_genotypes<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &SearchGenotypesFilter,
    limit: i64,
) -> Result<Vec<Genotype>, RepositoryError> {
    let rows = sqlx::query!(
        r#"
            SELECT
                g.id,
                g.generated_at,
                g.type_name,
                g.type_hash,
                g.genome,
                g.genome_hash,
                g.request_id,
                g.generation_id,
                g.parent_a,
                g.parent_b
            FROM fx_durable_ga.genotypes g
            WHERE (
                $1::UUID[] IS NULL OR g.request_id = ANY($1)
            )
            AND (
                $2::INTEGER[] IS NULL OR g.generation_id = ANY($2)
            )
            AND (
                $3::UUID[] IS NULL OR g.id = ANY($3)
            )
            AND (
                $4::UUID IS NULL OR g.id > $4::UUID
            )
            AND ($5::TEXT IS NULL OR g.id::TEXT ILIKE $5)
            ORDER BY
                CASE WHEN $6 THEN RANDOM() ELSE NULL END NULLS LAST,
                g.id ASC
            LIMIT $7;
        "#,
        filter.request_ids.as_deref(),
        filter.generation_ids.as_deref(),
        filter.genotype_ids.as_deref(),
        filter.cursor,
        filter.search.as_ref().map(|q| format!("%{}%", q)),
        filter.order_random,
        limit,
    )
    .fetch_all(tx)
    .await?;

    let genotypes = rows
        .into_iter()
        .map(|row| Genotype {
            id: row.id,
            generated_at: row.generated_at,
            type_name: row.type_name,
            type_hash: row.type_hash,
            genome: row.genome,
            genome_hash: row.genome_hash,
            request_id: row.request_id,
            generation_id: row.generation_id,
            parent_a: row.parent_a,
            parent_b: row.parent_b,
        })
        .collect();

    Ok(genotypes)
}

#[cfg(test)]
mod tests_search_genotypes {
    use super::super::test_tools::seed;
    use super::{SearchGenotypesFilter, search_genotypes};
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_searches_genotypes_with_request_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (rid_1, _, gids) = seed(&pool).await;

        let found = search_genotypes(
            &pool,
            &SearchGenotypesFilter::default().with_request_id(rid_1),
            5,
        )
        .await?;

        let actual: Vec<Uuid> = found.iter().map(|g| g.id).collect();
        assert_eq!(vec![gids[0], gids[1]], actual);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_searches_genotypes_with_generation_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (.., gids) = seed(&pool).await;

        let found = search_genotypes(
            &pool,
            &SearchGenotypesFilter::default().with_generation_id(2),
            5,
        )
        .await?;

        let actual: Vec<Uuid> = found.iter().map(|g| g.id).collect();
        assert_eq!(vec![gids[1], gids[4]], actual);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_paginates_with_cursor(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (rid_1, _rid_2, _gids) = seed(&pool).await;

        let first_page = search_genotypes(
            &pool,
            &SearchGenotypesFilter::default().with_request_id(rid_1),
            1,
        )
        .await?;

        assert_eq!(first_page.len(), 1);
        let cursor = first_page[0].id;

        let next_page = search_genotypes(
            &pool,
            &SearchGenotypesFilter::default()
                .with_request_id(rid_1)
                .with_cursor(cursor),
            10,
        )
        .await?;

        assert!(next_page.iter().all(|g| g.id() > cursor));

        Ok(())
    }
}
