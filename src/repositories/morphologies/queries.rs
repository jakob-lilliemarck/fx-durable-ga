use super::Error;
use super::models::DBMorphology;
use crate::models::Morphology;
use sqlx::PgExecutor;
use tracing::instrument;

/// Inserts a new morphology into the database.
/// Returns the inserted morphology with database-generated fields.
#[instrument(level = "debug", skip(tx), fields(type_name = %morphology.type_name, type_hash = morphology.type_hash))]
pub(crate) async fn new_morphology<'tx, E: PgExecutor<'tx>>(
    tx: E,
    morphology: Morphology,
) -> Result<Morphology, Error> {
    let db_morphology = DBMorphology::try_from(morphology)?;
    let db_morphology = sqlx::query_as!(
        DBMorphology,
        r#"
            INSERT INTO fx_durable_ga.morphologies (
                revised_at,
                type_name,
                type_hash,
                gene_bounds
            )
            VALUES ($1, $2, $3, $4)
            RETURNING
                revised_at,
                type_name,
                type_hash,
                gene_bounds;
            "#,
        db_morphology.revised_at,
        db_morphology.type_name,
        db_morphology.type_hash,
        db_morphology.gene_bounds
    )
    .fetch_one(tx)
    .await?;

    let morphology = Morphology::try_from(db_morphology)?;
    Ok(morphology)
}

#[cfg(test)]
mod new_morphology_tests {
    use super::*;
    use crate::models::GeneBounds;
    use chrono::SubsecRound;

    #[sqlx::test(migrations = false)]
    async fn it_inserts_a_new_morphology(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::integer(1, 10, 10)?,
                GeneBounds::integer(1, 10, 10)?,
                GeneBounds::integer(1, 10, 10)?,
            ],
        );
        let morphology_clone = morphology.clone();

        let inserted = new_morphology(&pool, morphology).await?;

        assert_eq!(
            morphology_clone.revised_at.trunc_subsecs(6),
            inserted.revised_at
        );
        assert_eq!(morphology_clone.type_name, inserted.type_name);
        assert_eq!(morphology_clone.type_hash, inserted.type_hash);
        assert_eq!(morphology_clone.gene_bounds, inserted.gene_bounds);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::integer(1, 10, 10)?,
                GeneBounds::integer(1, 10, 10)?,
                GeneBounds::integer(1, 10, 10)?,
            ],
        );
        let morphology_clone = morphology.clone();

        new_morphology(&pool, morphology).await?;
        let inserted = new_morphology(&pool, morphology_clone).await;

        assert!(inserted.is_err());

        Ok(())
    }
}

/// Retrieves a morphology by its type hash.
/// Returns NotFound error if the morphology doesn't exist.
#[instrument(level = "debug", skip(tx), fields(type_hash = type_hash))]
pub(crate) async fn get_morphology<'tx, E: PgExecutor<'tx>>(
    tx: E,
    type_hash: i32,
) -> Result<Morphology, Error> {
    let db_morphology = sqlx::query_as!(
        DBMorphology,
        r#"
            SELECT
                type_name,
                type_hash,
                revised_at,
                gene_bounds
            FROM fx_durable_ga.morphologies
            WHERE type_hash = $1;
        "#,
        type_hash
    )
    .fetch_one(tx)
    .await
    .map_err(|err| match err {
        sqlx::Error::RowNotFound => Error::NotFound,
        err => Error::Database(err),
    })?;

    let morphology = Morphology::try_from(db_morphology)?;
    Ok(morphology)
}

#[cfg(test)]
mod get_morphology_tests {
    use super::*;
    use crate::models::GeneBounds;

    #[sqlx::test(migrations = false)]
    async fn it_gets_an_existing_morphology(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::integer(1, 10, 10)?,
                GeneBounds::integer(1, 10, 10)?,
                GeneBounds::integer(1, 10, 10)?,
            ],
        );
        let morphology_type_hash = morphology.type_hash;

        let _ = new_morphology(&pool, morphology).await?;
        let selected = get_morphology(&pool, morphology_type_hash).await?;

        assert_eq!(morphology_type_hash, selected.type_hash);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::integer(1, 10, 10)?,
                GeneBounds::integer(1, 10, 10)?,
                GeneBounds::integer(1, 10, 10)?,
            ],
        );
        let morphology_type_hash = morphology.type_hash;

        let selected = get_morphology(&pool, morphology_type_hash).await;

        assert!(selected.is_err());
        assert!(matches!(selected, Err(Error::NotFound)));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_handles_database_errors(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        pool.close().await;
        let result = get_morphology(&pool, 1).await;
        assert!(matches!(result, Err(Error::Database(_))));
        Ok(())
    }
}
