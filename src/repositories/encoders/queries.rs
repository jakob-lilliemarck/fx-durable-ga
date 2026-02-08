use crate::repositories::encoders::Encoder;
use sqlx::PgExecutor;
use uuid::Uuid;

pub(crate) async fn get_encoder<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: &Uuid,
) -> Result<Option<Encoder>, super::Error> {
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
    .fetch_optional(tx)
    .await?;

    Ok(encoder)
}

#[cfg(test)]
mod tests_get {
    use super::{get_encoder, store_encoder, Encoder};
    use chrono::{TimeZone, Utc};
    use serde_json::json;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_gets_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let trained_at = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let encoder = Encoder {
            id: Uuid::now_v7(),
            model_type: "lstm".to_string(),
            model_config: json!({ "layers": 2 }),
            model_weights: vec![1, 2, 3, 4],
            model_format: "bin".to_string(),
            shape_in: vec![128, 256],
            shape_out: 64,
            trained_at,
            trained_on_checksum: vec![9, 8, 7, 6],
        };

        let stored = store_encoder(&pool, &encoder).await?;
        let fetched = get_encoder(&pool, &stored.id).await?;

        let fetched = fetched.expect("expected encoder");
        assert_eq!(stored.id, fetched.id);
        assert_eq!(stored.model_type, fetched.model_type);
        assert_eq!(stored.model_config, fetched.model_config);
        assert_eq!(stored.model_weights, fetched.model_weights);
        assert_eq!(stored.shape_in, fetched.shape_in);
        assert_eq!(stored.shape_out, fetched.shape_out);
        assert_eq!(stored.trained_at, fetched.trained_at);
        assert_eq!(stored.trained_on_checksum, fetched.trained_on_checksum);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_none_for_missing_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let missing_id = Uuid::now_v7();
        let encoder = get_encoder(&pool, &missing_id).await?;

        assert!(encoder.is_none());
        Ok(())
    }
}

pub(crate) async fn store_encoder<'tx, E: PgExecutor<'tx>>(
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
    use super::{store_encoder, Encoder};
    use chrono::{TimeZone, Utc};
    use serde_json::json;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_stores_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let trained_at = Utc.with_ymd_and_hms(2023, 12, 1, 12, 0, 0).unwrap();
        let encoder = Encoder {
            id: Uuid::now_v7(),
            model_type: "transformer".to_string(),
            model_config: json!({ "heads": 4 }),
            model_weights: vec![5, 4, 3, 2, 1],
            model_format: "bin".to_string(),
            shape_in: vec![64],
            shape_out: 32,
            trained_at,
            trained_on_checksum: vec![1, 2, 3, 4],
        };

        let stored = store_encoder(&pool, &encoder).await?;

        assert_eq!(stored.id, encoder.id);
        assert_eq!(stored.model_type, encoder.model_type);
        assert_eq!(stored.model_config, encoder.model_config);
        assert_eq!(stored.model_weights, encoder.model_weights);
        assert_eq!(stored.model_format, encoder.model_format);
        assert_eq!(stored.shape_in, encoder.shape_in);
        assert_eq!(stored.shape_out, encoder.shape_out);
        assert_eq!(stored.trained_at, encoder.trained_at);
        assert_eq!(stored.trained_on_checksum, encoder.trained_on_checksum);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_duplicate_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let trained_at = Utc.with_ymd_and_hms(2023, 11, 15, 6, 0, 0).unwrap();
        let encoder = Encoder {
            id: Uuid::now_v7(),
            model_type: "cnn".to_string(),
            model_config: json!({ "filters": 16 }),
            model_weights: vec![0, 1, 0, 1],
            model_format: "bin".to_string(),
            shape_in: vec![32, 32, 3],
            shape_out: 10,
            trained_at,
            trained_on_checksum: vec![42, 42, 42],
        };

        store_encoder(&pool, &encoder).await?;
        let result = store_encoder(&pool, &encoder).await;

        assert!(result.is_err());
        Ok(())
    }
}
