use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sqlx::{
    Decode, Encode, PgExecutor, Postgres, Type,
    encode::IsNull,
    postgres::{PgArgumentBuffer, PgTypeInfo, PgValueRef},
};
use tracing::instrument;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid digest: {0}")]
    InvalidDigest(String),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Digest([u8; 64]);

impl std::fmt::Debug for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self, f)
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Serialize for Digest {
    #[instrument(level = "debug", skip_all)]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Digest {
    #[instrument(level = "debug", skip(deserializer))]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Digest::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

impl Digest {
    #[instrument(level = "debug")]
    pub fn from_hex(hex: &str) -> Result<Self, Error> {
        if hex.len() != 64 || !hex.is_ascii() {
            return Err(Error::InvalidDigest(hex.to_string()));
        }

        let mut bytes = [0u8; 64];
        bytes.copy_from_slice(hex.as_bytes());
        Ok(Digest(bytes))
    }

    #[instrument(level = "debug", skip(self))]
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).unwrap()
    }
}

impl From<String> for Digest {
    #[instrument(level = "debug")]
    fn from(value: String) -> Self {
        Digest::from_hex(&value).unwrap()
    }
}

impl Type<Postgres> for Digest {
    fn type_info() -> PgTypeInfo {
        <String as Type<Postgres>>::type_info()
    }
}

impl<'r> Decode<'r, Postgres> for Digest {
    #[instrument(level = "debug", skip(value))]
    fn decode(value: PgValueRef<'r>) -> Result<Self, sqlx::error::BoxDynError> {
        let s = <&str as Decode<Postgres>>::decode(value)?;
        Digest::from_hex(s).map_err(|e| e.into())
    }
}

impl Encode<'_, Postgres> for Digest {
    #[instrument(level = "debug", skip_all)]
    fn encode_by_ref(
        &self,
        buf: &mut PgArgumentBuffer,
    ) -> Result<IsNull, sqlx::error::BoxDynError> {
        <&str as Encode<Postgres>>::encode(self.as_str(), buf)
    }
}

#[derive(Debug)]
pub struct Encoder {
    pub(crate) digest: Digest,
    pub(crate) encodable_type_name: String,
    pub(crate) model_config: serde_json::Value,
    pub(crate) model_weights: Vec<u8>,
    pub(crate) model_format: String,
    pub(crate) shape_in: Vec<i32>,
    pub(crate) shape_out: i32,
}

/// The availability state of an encoder
/// NotFound:       it is not known to the system
/// NotAvailable:   it is known to the system but can not yet be used
/// Available:      it is known to the system and it is ready to use
#[derive(Debug, PartialEq)]
pub(crate) enum EncoderAvailability {
    NotFound,
    NotAvailable,
    Available,
}

impl From<Option<bool>> for EncoderAvailability {
    fn from(value: Option<bool>) -> Self {
        match value {
            Some(true) => EncoderAvailability::Available,
            Some(false) => EncoderAvailability::NotAvailable,
            None => Self::NotFound,
        }
    }
}

/// Get an encoder by id
#[instrument(level = "debug", skip(tx))]
pub async fn get_encoder<'tx, E: PgExecutor<'tx>>(
    tx: E,
    digest: &Digest,
) -> Result<Option<Encoder>, super::Error> {
    let encoder = sqlx::query_as!(
        Encoder,
        r#"
            SELECT
                digest,
                encodable_type_name,
                model_config,
                model_weights,
                model_format,
                shape_in,
                shape_out
            FROM encoders
            WHERE digest = $1;
        "#,
        digest.as_str(),
    )
    .fetch_optional(tx)
    .await?;

    Ok(encoder)
}

#[cfg(test)]
mod tests_get {
    use super::super::{Digest, Encoder};
    use super::{get_encoder, store_encoder};
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    #[sqlx::test(migrations = false)]
    async fn it_gets_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let trained_at = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();

        let encoder = Encoder {
            digest: digest,
            model_config: json!({ "layers": 2 }),
            model_weights: vec![1, 2, 3, 4],
            model_format: "bin".to_string(),
            shape_in: vec![128, 256],
            shape_out: 64,
            encodable_type_name: "encodable_type_name".to_string(),
        };

        let stored = store_encoder(&pool, &encoder, &trained_at).await?;

        let fetched = get_encoder(&pool, &encoder.digest)
            .await?
            .expect("Expect an encoder to be returned");

        assert_eq!(stored.digest, fetched.digest);
        assert_eq!(stored.model_config, fetched.model_config);
        assert_eq!(stored.model_weights, fetched.model_weights);
        assert_eq!(stored.model_format, fetched.model_format);
        assert_eq!(stored.shape_in, fetched.shape_in);
        assert_eq!(stored.shape_out, fetched.shape_out);

        Ok(())
    }
}

/// Store an encoder
#[instrument(
    level = "debug",
    skip(tx, encoder),
    fields(type_name = %encoder.encodable_type_name, digest = %encoder.digest))]
pub(crate) async fn store_encoder<'tx, E: PgExecutor<'tx>>(
    tx: E,
    encoder: &Encoder,
    trained_at: &DateTime<Utc>,
) -> Result<Encoder, super::Error> {
    let encoder = sqlx::query_as!(
        Encoder,
        r#"
            INSERT INTO fx_durable_ga.encoders (
                digest,
                encodable_type_name,
                model_config,
                model_weights,
                model_format,
                shape_in,
                shape_out,
                trained_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            RETURNING
                digest,
                encodable_type_name,
                model_config,
                model_weights,
                model_format,
                shape_in,
                shape_out;
        "#,
        encoder.digest.as_str(),
        encoder.encodable_type_name,
        encoder.model_config,
        encoder.model_weights,
        encoder.model_format,
        &encoder.shape_in,
        encoder.shape_out,
        trained_at
    )
    .fetch_one(tx)
    .await?;

    Ok(encoder)
}

#[cfg(test)]
mod tests_store {
    use super::super::Digest;
    use super::{Encoder, store_encoder};
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    #[sqlx::test(migrations = false)]
    async fn it_stores_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let trained_at = Utc.with_ymd_and_hms(2023, 12, 1, 12, 0, 0).unwrap();
        let digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();

        let encoder = Encoder {
            digest,
            encodable_type_name: "encodable_type_name".to_string(),
            model_config: json!({ "heads": 4 }),
            model_weights: vec![5, 4, 3, 2, 1],
            model_format: "bin".to_string(),
            shape_in: vec![64],
            shape_out: 32,
        };

        let stored = store_encoder(&pool, &encoder, &trained_at).await?;

        assert_eq!(stored.digest, encoder.digest);
        assert_eq!(stored.encodable_type_name, encoder.encodable_type_name);
        assert_eq!(stored.model_config, encoder.model_config);
        assert_eq!(stored.model_weights, encoder.model_weights);
        assert_eq!(stored.model_format, encoder.model_format);
        assert_eq!(stored.shape_in, encoder.shape_in);
        assert_eq!(stored.shape_out, encoder.shape_out);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_duplicate_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let trained_at = Utc.with_ymd_and_hms(2023, 11, 15, 6, 0, 0).unwrap();
        let digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();

        let encoder = Encoder {
            digest,
            encodable_type_name: "encodable_type_name".to_string(),
            model_config: json!({ "filters": 16 }),
            model_weights: vec![0, 1, 0, 1],
            model_format: "bin".to_string(),
            shape_in: vec![32, 32, 3],
            shape_out: 10,
        };

        store_encoder(&pool, &encoder, &trained_at).await?;
        let result = store_encoder(&pool, &encoder, &trained_at).await;

        assert!(result.is_err());
        Ok(())
    }
}

#[cfg(test)]
mod test_tools {
    use super::super::{Digest, Encoder};
    use chrono::{TimeZone, Utc};
    use sqlx::PgExecutor;

    pub async fn seed_encoder<'tx, E: PgExecutor<'tx>>(tx: E, c: char) -> anyhow::Result<Encoder> {
        let trained_at = Utc.with_ymd_and_hms(2023, 12, 1, 12, 0, 0).unwrap();
        let digest = Digest::from_hex(&format!(
            "000000000000000000000000000000000000000000000000000000000000000{}",
            c
        ))
        .unwrap();

        let encoder = Encoder {
            digest,
            encodable_type_name: "encodable_type_name".to_string(),
            model_config: serde_json::json!({ "heads": 4 }),
            model_weights: vec![5, 4, 3, 2, 1],
            model_format: "bin".to_string(),
            shape_in: vec![64],
            shape_out: 32,
        };

        let stored = super::store_encoder(tx, &encoder, &trained_at).await?;
        Ok(stored)
    }
}

// Get digests of enabled encoders
#[instrument(level = "debug", skip(tx))]
pub async fn get_available_encoder_digests<'tx, E: PgExecutor<'tx>>(
    tx: E,
    digests: &[Digest],
) -> Result<Vec<Digest>, super::Error> {
    let hexes: Vec<&str> = digests.iter().map(|d| d.as_str()).collect();
    let available = sqlx::query_scalar::<_, Digest>(
        r#"
        SELECT digest::TEXT
        FROM fx_durable_ga.available_encoder_digests
        WHERE digest = ANY($1);
        "#,
    )
    .bind(&hexes)
    .fetch_all(tx)
    .await?;
    Ok(available)
}

#[cfg(test)]
mod tests_get_available_encoder_digests {
    use super::get_available_encoder_digests;
    use crate::services::indexing::encoder_queries::store_encoder_availability;
    use chrono::Utc;

    #[sqlx::test(migrations = false)]
    async fn it_gets_enabled_encoder_pairings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let encoder_a = super::test_tools::seed_encoder(&pool, 'a').await?;
        let encoder_b = super::test_tools::seed_encoder(&pool, 'b').await?;

        store_encoder_availability(&pool, &encoder_a.digest, true, &Utc::now()).await?;
        store_encoder_availability(&pool, &encoder_b.digest, true, &Utc::now()).await?;

        let enabled =
            get_available_encoder_digests(&pool, &[encoder_a.digest, encoder_b.digest]).await?;

        assert_eq!(enabled.len(), 2);
        assert!(enabled.iter().any(|digest| digest == &encoder_a.digest));
        assert!(enabled.iter().any(|digest| digest == &encoder_b.digest));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_ignores_disabled_encoder_pairings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // just enabled
        let encoder_a = super::test_tools::seed_encoder(&pool, 'a').await?;
        store_encoder_availability(&pool, &encoder_a.digest, true, &Utc::now()).await?;

        // just disabled
        let encoder_b = super::test_tools::seed_encoder(&pool, 'b').await?;
        store_encoder_availability(&pool, &encoder_b.digest, false, &Utc::now()).await?;

        // enabled then disabled
        let encoder_c = super::test_tools::seed_encoder(&pool, 'c').await?;
        store_encoder_availability(&pool, &encoder_c.digest, true, &Utc::now()).await?;
        store_encoder_availability(&pool, &encoder_c.digest, false, &Utc::now()).await?;

        // disabled then enabled
        let encoder_d = super::test_tools::seed_encoder(&pool, 'd').await?;
        store_encoder_availability(&pool, &encoder_d.digest, false, &Utc::now()).await?;
        store_encoder_availability(&pool, &encoder_d.digest, true, &Utc::now()).await?;

        let enabled = get_available_encoder_digests(
            &pool,
            &[
                encoder_a.digest,
                encoder_b.digest,
                encoder_c.digest,
                encoder_d.digest,
            ],
        )
        .await?;

        assert_eq!(enabled.len(), 2);

        // enabled
        assert!(enabled.iter().any(|digest| *digest == encoder_a.digest));
        assert!(enabled.iter().any(|digest| *digest == encoder_d.digest));

        // disabled
        assert!(enabled.iter().all(|digest| *digest != encoder_b.digest));
        assert!(enabled.iter().all(|digest| *digest != encoder_c.digest));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_ignores_noncurrent_pairings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let encoder_a = super::test_tools::seed_encoder(&pool, 'a').await?;
        store_encoder_availability(&pool, &encoder_a.digest, true, &Utc::now()).await?;
        store_encoder_availability(&pool, &encoder_a.digest, false, &Utc::now()).await?;

        let encoder_b = super::test_tools::seed_encoder(&pool, 'b').await?;
        store_encoder_availability(&pool, &encoder_b.digest, false, &Utc::now()).await?;
        store_encoder_availability(&pool, &encoder_b.digest, true, &Utc::now()).await?;

        let enabled =
            get_available_encoder_digests(&pool, &[encoder_a.digest, encoder_b.digest]).await?;

        assert_eq!(enabled.len(), 1);
        assert_eq!(enabled[0], encoder_b.digest);

        Ok(())
    }
}

// Store encoder state
#[instrument(level = "debug", skip(tx))]
pub(crate) async fn store_encoder_availability<'tx, E: PgExecutor<'tx>>(
    tx: E,
    indexer_id: &Digest,
    is_available: bool,
    revised_at: &DateTime<Utc>,
) -> Result<(EncoderAvailability, bool), super::Error> {
    let record = sqlx::query!(
        r#"
        WITH last_state AS (
            SELECT DISTINCT ON (digest) is_available
            FROM encoder_availability
            WHERE digest = $1::CHAR(64)
            ORDER BY digest, revised_at DESC
        ),
        inserted AS (
            INSERT INTO encoder_availability (digest, is_available, revised_at)
            SELECT $1, $2, $3
            WHERE NOT EXISTS (
                SELECT 1 FROM last_state WHERE is_available = $2
            )
            RETURNING is_available
        )
        SELECT
            is_available AS "is_available!:bool",
            TRUE AS "was_changed!:bool"
        FROM inserted
        UNION ALL
        SELECT
            is_available AS "is_available!:bool",
            FALSE AS "was_changed!:bool"
        FROM last_state
        WHERE NOT EXISTS (SELECT 1 FROM inserted);
        "#,
        indexer_id.as_str(),
        is_available,
        revised_at
    )
    .fetch_one(tx)
    .await?;

    Ok((Some(record.is_available).into(), record.was_changed))
}

#[cfg(test)]
mod tests_store_encoder_state {
    use super::super::Digest;
    use super::{EncoderAvailability, store_encoder_availability};
    use chrono::{Duration, TimeZone, Utc};

    #[sqlx::test(migrations = false)]
    async fn it_stores_state_of_unseen_digests(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000100")?;
        let revised_at = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();

        let (state, was_changed) =
            store_encoder_availability(&pool, &digest, true, &revised_at).await?;

        assert!(matches!(state, EncoderAvailability::Available));
        assert!(was_changed);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_writes_a_record_if_state_differ(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000101")?;
        let first_revision = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let later_revision = first_revision + Duration::minutes(5);

        let (initial_state, initial_changed) =
            store_encoder_availability(&pool, &digest, false, &first_revision).await?;
        assert!(matches!(initial_state, EncoderAvailability::NotAvailable));
        assert!(initial_changed);

        let (next_state, next_changed) =
            store_encoder_availability(&pool, &digest, true, &later_revision).await?;
        assert!(matches!(next_state, EncoderAvailability::Available));
        assert!(next_changed);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_last_record_if_state_is_same(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000102")?;
        let first_revision = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let later_revision = first_revision + Duration::minutes(10);

        let (initial_state, initial_changed) =
            store_encoder_availability(&pool, &digest, false, &first_revision).await?;
        assert!(matches!(initial_state, EncoderAvailability::NotAvailable));
        assert!(initial_changed);

        let (subsequent_state, subsequent_changed) =
            store_encoder_availability(&pool, &digest, false, &later_revision).await?;
        assert!(matches!(
            subsequent_state,
            EncoderAvailability::NotAvailable
        ));
        assert!(!subsequent_changed);

        Ok(())
    }
}

// Get encoder state
#[instrument(level = "debug", skip(tx))]
pub(crate) async fn get_encoder_availability<'tx, E: PgExecutor<'tx>>(
    tx: E,
    indexer_id: &Digest,
) -> Result<EncoderAvailability, super::Error> {
    let scalar = sqlx::query_scalar!(
        r#"
        SELECT is_available "is_available!: bool"
        FROM fx_durable_ga.latest_encoder_availability
        WHERE digest = $1::CHAR(64)
        "#,
        indexer_id.as_str()
    )
    .fetch_optional(tx)
    .await?;

    Ok(scalar.into())
}

#[cfg(test)]
mod tests_get_encoder_availability {
    use super::super::Digest;
    use super::{EncoderAvailability, get_encoder_availability, store_encoder_availability};
    use chrono::{Duration, TimeZone, Utc};

    #[sqlx::test(migrations = false)]
    async fn it_gets_state_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let state = get_encoder_availability(&pool, &data.missing_digest).await?;
        assert!(matches!(state, EncoderAvailability::NotFound));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_state_not_ready(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let state = get_encoder_availability(&pool, &data.not_ready_digest).await?;
        assert!(matches!(state, EncoderAvailability::NotAvailable));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_state_ready(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let state = get_encoder_availability(&pool, &data.ready_digest).await?;
        assert!(matches!(state, EncoderAvailability::Available));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_the_last_of_multiple_records(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let state = get_encoder_availability(&pool, &data.multi_record_digest).await?;
        assert!(matches!(state, EncoderAvailability::Available));

        Ok(())
    }

    struct SeedData {
        ready_digest: Digest,
        not_ready_digest: Digest,
        multi_record_digest: Digest,
        missing_digest: Digest,
    }

    async fn seed(pool: &sqlx::PgPool) -> anyhow::Result<SeedData> {
        let ready_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000200")?;
        let not_ready_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000201")?;
        let multi_record_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000202")?;
        let missing_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000203")?;

        let base_time = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();

        store_encoder_availability(pool, &ready_digest, true, &base_time).await?;
        store_encoder_availability(
            pool,
            &not_ready_digest,
            false,
            &(base_time + Duration::minutes(1)),
        )
        .await?;
        store_encoder_availability(
            pool,
            &multi_record_digest,
            false,
            &(base_time + Duration::minutes(2)),
        )
        .await?;
        store_encoder_availability(
            pool,
            &multi_record_digest,
            true,
            &(base_time + Duration::minutes(3)),
        )
        .await?;

        Ok(SeedData {
            ready_digest,
            not_ready_digest,
            multi_record_digest,
            missing_digest,
        })
    }
}
