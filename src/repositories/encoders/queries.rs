use crate::repositories::encoders::Encoder;
use sqlx::PgExecutor;
use uuid::Uuid;

pub(crate) async fn get<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: &Uuid,
) -> Result<Encoder, super::Error> {
    let encoder = sqlx::query_as!(
        Encoder,
        r#"
            SELECT
                id,
                model_type,
                model_config,
                model_weights,
                model_format,
                shape_in,
                shape_out,
                trained_at,
                trained_on_checksum
            FROM fx_durable_ga.encoders
            WHERE id = $1;
        "#,
        id
    )
    .fetch_one(tx)
    .await?;

    Ok(encoder)
}

#[cfg(test)]
mod tests_get {
    #[sqlx::test(migrations = false)]
    async fn it_gets_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        todo!()
    }
}

pub(crate) async fn store<'tx, E: PgExecutor<'tx>>(
    tx: E,
    encoder: &Encoder,
) -> Result<Encoder, super::Error> {
    let encoder = sqlx::query_as!(
        Encoder,
        r#"
            INSERT INTO fx_durable_ga.encoders (
                id,
                model_type,
                model_config,
                model_weights,
                model_format,
                shape_in,
                shape_out,
                trained_at,
                trained_on_checksum
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            RETURNING
                id,
                model_type,
                model_config,
                model_weights,
                model_format,
                shape_in,
                shape_out,
                trained_at,
                trained_on_checksum
        "#,
        encoder.id,
        encoder.model_type,
        encoder.model_config,
        encoder.model_weights,
        encoder.model_format,
        &encoder.shape_in,
        encoder.shape_out,
        encoder.trained_at,
        encoder.trained_on_checksum,
    )
    .fetch_one(tx)
    .await?;

    Ok(encoder)
}

#[cfg(test)]
mod tests_store {
    #[sqlx::test(migrations = false)]
    async fn it_stores_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        todo!()
    }
}
