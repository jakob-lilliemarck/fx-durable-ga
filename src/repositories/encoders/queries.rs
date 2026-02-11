use super::EncoderPairing;
use crate::repositories::encoders::Encoder;
use chrono::Utc;
use sqlx::PgExecutor;
use uuid::Uuid;

pub(crate) struct TogglingResult {
    pub(crate) is_enabled: bool,
    pub(crate) was_changed: bool,
}

/// Get the latest encoder for a given type hash
pub(crate) async fn get_encoder<'tx, E: PgExecutor<'tx>>(
    tx: E,
    encoder_id: &Uuid,
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
            FROM encoders
            WHERE id = $1;
        "#,
        encoder_id
    )
    .fetch_optional(tx)
    .await?;

    Ok(encoder)
}

#[cfg(test)]
mod tests_get {
    use super::{Encoder, get_encoder, store_encoder};
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

        let fetched = get_encoder(&pool, &stored.id)
            .await?
            .expect("Expect an encoder to be returned");

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
}

pub(crate) async fn toggle_encoder_pairing<'tx, E: PgExecutor<'tx>>(
    tx: E,
    type_hash: i32,
    encoder_id: &Uuid,
    is_enabled: bool,
) -> Result<TogglingResult, super::Error> {
    let timestamp = Utc::now();

    let result = sqlx::query_as!(
        TogglingResult,
        r#"
            WITH last_state AS (
                SELECT is_enabled
                FROM fx_durable_ga.encoder_toggles
                WHERE type_hash = $1 AND encoder_id = $2
                ORDER BY timestamp DESC
                LIMIT 1
            ),
            inserted AS (
                INSERT INTO fx_durable_ga.encoder_toggles (
                    type_hash,
                    encoder_id,
                    is_enabled,
                    timestamp
                )
                SELECT $1, $2, $3, $4
                WHERE NOT EXISTS (
                    SELECT 1 FROM last_state WHERE is_enabled = $3
                )
                RETURNING is_enabled
            )
            SELECT
                is_enabled AS "is_enabled!:bool",
                TRUE AS "was_changed!:bool"
            FROM inserted
            UNION ALL
            SELECT
                is_enabled AS "is_enabled!:bool",
                FALSE AS "was_changed!:bool"
            FROM last_state
            WHERE NOT EXISTS (SELECT 1 FROM inserted);
        "#,
        type_hash,
        encoder_id,
        is_enabled,
        timestamp
    )
    .fetch_one(tx)
    .await?;

    Ok(result)
}

#[cfg(test)]
mod tests_toggle_encoder_pairing {
    use super::{get_encoder_pairings, toggle_encoder_pairing};
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_toggles_pairing_on(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let type_hash = 101;
        let encoder_id = Uuid::now_v7();

        let result = toggle_encoder_pairing(&pool, type_hash, &encoder_id, true).await?;
        assert!(result.is_enabled);
        assert!(result.was_changed);

        let pairings = get_encoder_pairings(&pool, &[type_hash]).await?;

        assert_eq!(pairings.len(), 1);
        assert_eq!(pairings[0].encoder_id, encoder_id);
        assert_eq!(pairings[0].type_hash, type_hash);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_toggles_pairing_off(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let type_hash = 102;
        let encoder_id = Uuid::now_v7();

        let enabled = toggle_encoder_pairing(&pool, type_hash, &encoder_id, true).await?;
        assert!(enabled.is_enabled);
        assert!(enabled.was_changed);

        let disabled = toggle_encoder_pairing(&pool, type_hash, &encoder_id, false).await?;
        assert!(!disabled.is_enabled);
        assert!(disabled.was_changed);

        let pairings = get_encoder_pairings(&pool, &[type_hash]).await?;
        assert!(pairings.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_does_not_duplicate_toggle_on(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let type_hash = 103;
        let encoder_id = Uuid::now_v7();

        let first_toggle = toggle_encoder_pairing(&pool, type_hash, &encoder_id, true).await?;
        assert!(first_toggle.is_enabled);
        assert!(first_toggle.was_changed);

        let second_toggle = toggle_encoder_pairing(&pool, type_hash, &encoder_id, true).await?;
        assert!(second_toggle.is_enabled);
        assert!(!second_toggle.was_changed);

        let pairings = get_encoder_pairings(&pool, &[type_hash]).await?;
        assert_eq!(pairings.len(), 1);
        assert_eq!(pairings[0].encoder_id, encoder_id);
        assert_eq!(pairings[0].type_hash, type_hash);

        let toggle_count = sqlx::query_scalar!(
            r#"
                SELECT COUNT(*)::BIGINT AS "count!:i64"
                FROM fx_durable_ga.encoder_toggles
                WHERE type_hash = $1 AND encoder_id = $2
            "#,
            type_hash,
            encoder_id
        )
        .fetch_one(&pool)
        .await?;

        assert_eq!(toggle_count, 1);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_does_not_duplicate_toggle_off(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let type_hash = 104;
        let encoder_id = Uuid::now_v7();

        let first_enable = toggle_encoder_pairing(&pool, type_hash, &encoder_id, true).await?;
        assert!(first_enable.is_enabled);
        assert!(first_enable.was_changed);

        let first_disable = toggle_encoder_pairing(&pool, type_hash, &encoder_id, false).await?;
        assert!(!first_disable.is_enabled);
        assert!(first_disable.was_changed);

        let second_disable = toggle_encoder_pairing(&pool, type_hash, &encoder_id, false).await?;
        assert!(!second_disable.is_enabled);
        assert!(!second_disable.was_changed);

        let pairings = get_encoder_pairings(&pool, &[type_hash]).await?;
        assert!(pairings.is_empty());

        let toggle_count = sqlx::query_scalar!(
            r#"
                SELECT COUNT(*)::BIGINT AS "count!:i64"
                FROM fx_durable_ga.encoder_toggles
                WHERE type_hash = $1 AND encoder_id = $2
            "#,
            type_hash,
            encoder_id
        )
        .fetch_one(&pool)
        .await?;

        assert_eq!(toggle_count, 2);

        Ok(())
    }
}

pub(crate) async fn get_encoder_pairings<'tx, E: PgExecutor<'tx>>(
    tx: E,
    type_hashes: &[i32],
) -> Result<Vec<EncoderPairing>, super::Error> {
    let toggled = sqlx::query_as!(
        EncoderPairing,
        r#"
        SELECT encoder_id, type_hash
        FROM (
            SELECT DISTINCT ON (encoder_id, type_hash)
                encoder_id,
                type_hash,
                is_enabled
            FROM encoder_toggles
            WHERE type_hash = ANY($1)
            ORDER BY encoder_id, type_hash, timestamp DESC
        ) latest
        WHERE is_enabled = true;
        "#,
        type_hashes
    )
    .fetch_all(tx)
    .await?;

    Ok(toggled)
}

#[cfg(test)]
mod tests_get_toggled_encoder_ids {
    use super::{get_encoder_pairings, toggle_encoder_pairing};
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_gets_enabled_encoder_pairings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let encoder_a = Uuid::now_v7();
        let encoder_b = Uuid::now_v7();
        let type_hash_a = 201;
        let type_hash_b = 202;

        toggle_encoder_pairing(&pool, type_hash_a, &encoder_a, true).await?;
        toggle_encoder_pairing(&pool, type_hash_b, &encoder_b, true).await?;

        let pairings = get_encoder_pairings(&pool, &[type_hash_a, type_hash_b]).await?;

        assert_eq!(pairings.len(), 2);
        assert!(
            pairings
                .iter()
                .any(|p| p.encoder_id == encoder_a && p.type_hash == type_hash_a)
        );
        assert!(
            pairings
                .iter()
                .any(|p| p.encoder_id == encoder_b && p.type_hash == type_hash_b)
        );

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_ignores_disabled_encoder_pairings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let enabled_encoder = Uuid::now_v7();
        let enabled_type_hash = 301;
        toggle_encoder_pairing(&pool, enabled_type_hash, &enabled_encoder, true).await?;

        let currently_disabled_encoder = Uuid::now_v7();
        let currently_disabled_type_hash = 302;
        toggle_encoder_pairing(
            &pool,
            currently_disabled_type_hash,
            &currently_disabled_encoder,
            false,
        )
        .await?;

        let disabled_after_enabled_encoder = Uuid::now_v7();
        let disabled_after_enabled_type_hash = 304;
        toggle_encoder_pairing(
            &pool,
            disabled_after_enabled_type_hash,
            &disabled_after_enabled_encoder,
            true,
        )
        .await?;
        toggle_encoder_pairing(
            &pool,
            disabled_after_enabled_type_hash,
            &disabled_after_enabled_encoder,
            false,
        )
        .await?;

        let reenabled_encoder = Uuid::now_v7();
        let reenabled_type_hash = 303;
        toggle_encoder_pairing(&pool, reenabled_type_hash, &reenabled_encoder, false).await?;
        toggle_encoder_pairing(&pool, reenabled_type_hash, &reenabled_encoder, true).await?;

        let pairings = get_encoder_pairings(
            &pool,
            &[
                enabled_type_hash,
                currently_disabled_type_hash,
                disabled_after_enabled_type_hash,
                reenabled_type_hash,
            ],
        )
        .await?;

        assert_eq!(pairings.len(), 2);
        assert!(
            pairings
                .iter()
                .any(|p| p.encoder_id == enabled_encoder && p.type_hash == enabled_type_hash)
        );
        assert!(
            pairings
                .iter()
                .any(|p| p.encoder_id == reenabled_encoder && p.type_hash == reenabled_type_hash)
        );
        assert!(
            pairings
                .iter()
                .all(|p| p.type_hash != currently_disabled_type_hash)
        );
        assert!(
            pairings
                .iter()
                .all(|p| p.type_hash != disabled_after_enabled_type_hash)
        );

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_ignores_noncurrent_pairings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let disabled_pair_encoder = Uuid::now_v7();
        let disabled_pair_type_hash = 401;
        toggle_encoder_pairing(&pool, disabled_pair_type_hash, &disabled_pair_encoder, true)
            .await?;
        toggle_encoder_pairing(
            &pool,
            disabled_pair_type_hash,
            &disabled_pair_encoder,
            false,
        )
        .await?;

        let enabled_pair_encoder = Uuid::now_v7();
        let enabled_pair_type_hash = 402;
        toggle_encoder_pairing(&pool, enabled_pair_type_hash, &enabled_pair_encoder, false).await?;
        toggle_encoder_pairing(&pool, enabled_pair_type_hash, &enabled_pair_encoder, true).await?;

        let pairings =
            get_encoder_pairings(&pool, &[disabled_pair_type_hash, enabled_pair_type_hash]).await?;

        assert_eq!(pairings.len(), 1);
        assert_eq!(pairings[0].encoder_id, enabled_pair_encoder);
        assert_eq!(pairings[0].type_hash, enabled_pair_type_hash);

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
    use super::{Encoder, store_encoder};
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
