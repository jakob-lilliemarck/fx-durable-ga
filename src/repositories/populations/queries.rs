use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx), fields(individuals_count = individuals.len()))]
pub(crate) async fn add_to_population<'tx, E: PgExecutor<'tx>>(
    tx: E,
    individuals: &[(Uuid, Uuid)], // (request_id, genotype_id)
) -> Result<(), super::Error> {
    if individuals.is_empty() {
        return Ok(());
    }

    let mut query_builder =
        sqlx::QueryBuilder::new("INSERT INTO fx_durable_ga.populations (request_id, genotype_id) ");

    query_builder.push_values(individuals, |mut b, (request_id, genotype_id)| {
        b.push_bind(request_id).push_bind(genotype_id);
    });

    query_builder.push(" ON CONFLICT DO NOTHING");

    query_builder.build().execute(tx).await?;

    Ok(())
}

#[cfg(test)]
mod add_to_population_tests {
    #[sqlx::test(migrations = "./migrations")]
    async fn it_adds_to_the_population(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }
}

#[instrument(level = "debug", skip(tx), fields(request_id = %request_id, genotype_id = %genotype_id))]
pub(crate) async fn remove_from_population<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: &Uuid,
    genotype_id: &Uuid,
) -> Result<(), super::Error> {
    sqlx::query_scalar!(
        r#"
            DELETE FROM fx_durable_ga.populations
            WHERE request_id = $1 AND genotype_id = $2;
        "#,
        request_id,
        genotype_id
    )
    .execute(tx)
    .await?;

    Ok(())
}

#[cfg(test)]
mod remove_from_population_tests {
    #[sqlx::test(migrations = "./migrations")]
    async fn it_removes_from_the_population(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }
}

#[instrument(level = "debug", skip(tx), fields(request_id = %request_id))]
pub(crate) async fn get_population_count<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: &Uuid,
) -> Result<i64, super::Error> {
    let count = sqlx::query_scalar!(
        r#"
            SELECT COUNT(*) "count!:i64"
            FROM fx_durable_ga.populations
            WHERE request_id = $1
        "#,
        request_id
    )
    .fetch_one(tx)
    .await?;

    Ok(count)
}

#[cfg(test)]
mod get_population_count_tests {
    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_the_population_count(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }
}
