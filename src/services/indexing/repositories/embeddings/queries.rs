use crate::repositories::ordering::SortOrder;
use crate::services::indexing::Digest;
use chrono::{DateTime, Utc};
use const_fnv1a_hash::fnv1a_hash_str_32;
use pgvector::Vector;
use sqlx::{PgExecutor, Row};
use tracing::instrument;
use uuid::Uuid;

pub const EMBEDDING_SIZE: usize = 256;

pub type EmbeddingValue = [f32; EMBEDDING_SIZE];

#[derive(Debug)]
pub struct Embedding {
    pub embedding_id: Uuid,
    pub tags: Vec<String>,
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub struct EmbeddingNew {
    id: Uuid,
    encoder_digest: Digest,
    encoded_at: DateTime<Utc>,
    value: EmbeddingValue,
}

impl EmbeddingNew {
    pub fn new(encoder_id: Digest, encoded_at: DateTime<Utc>, value: EmbeddingValue) -> Self {
        Self {
            id: Uuid::now_v7(),
            encoder_digest: encoder_id,
            encoded_at,
            value,
        }
    }

    pub fn id(&self) -> &Uuid {
        &self.id
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct TagNew {
    id: Uuid,
    tag_name: String,
    embedding_id: Uuid,
    tagged_at: DateTime<Utc>,
}

impl TagNew {
    #[instrument(level = "debug")]
    pub fn new(tag_name: &str, embedding_id: Uuid, tagged_at: DateTime<Utc>) -> Self {
        Self {
            id: Uuid::now_v7(),
            tag_name: tag_name.to_owned(),
            embedding_id,
            tagged_at,
        }
    }
}

#[derive(Debug)]
pub struct SearchEmbeddingsFilter {
    tags: Option<Vec<String>>,
    encoder_digest: Option<Digest>,
    embedding_ids: Option<Vec<Uuid>>,
    cursor: Option<Uuid>,
}

impl Default for SearchEmbeddingsFilter {
    #[instrument(level = "debug")]
    fn default() -> Self {
        Self {
            tags: None,
            encoder_digest: None,
            embedding_ids: None,
            cursor: None,
        }
    }
}

impl SearchEmbeddingsFilter {
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.get_or_insert_with(Vec::new).push(tag.to_string());
        self
    }

    pub fn with_encoder_id(mut self, encoder_digest: &Digest) -> Self {
        self.encoder_digest = Some(*encoder_digest);
        self
    }

    pub fn with_cursor(mut self, cursor: &Uuid) -> Self {
        self.cursor = Some(*cursor);
        self
    }

    pub fn with_embedding_id(mut self, embedding_id: &Uuid) -> Self {
        self.embedding_ids
            .get_or_insert_with(Vec::new)
            .push(*embedding_id);
        self
    }
}

#[instrument(level = "debug", skip(tx))]
pub async fn search_embeddings<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &SearchEmbeddingsFilter,
    limit: i64,
) -> Result<Vec<Embedding>, super::Error> {
    let tag_hashes = filter.tags.as_ref().map(|tags| {
        tags.iter()
            .map(|tag| fnv1a_hash_str_32(tag) as i64)
            .collect::<Vec<i64>>()
    });

    let similar = sqlx::query_as!(
        Embedding,
        r#"
        SELECT
            te.id AS "embedding_id!: Uuid",
            ARRAY_AGG(tag_name ORDER BY te.tag_name asc) AS "tags!: Vec<String>"
        FROM fx_durable_ga.tagged_embeddings te
        WHERE
            ($1::CHAR(64) IS NULL OR te.encoder_digest = $1::CHAR(64))
            AND ($2::BIGINT[] IS NULL OR te.tag_hash = ANY($2::BIGINT[]))
            AND ($3::UUID[] IS NULL OR te.id = ANY($3::UUID[]))
            AND ($4::UUID IS NULL OR te.id > $4::UUID)
        GROUP BY te.id
        ORDER BY te.id
        LIMIT $5;
        "#,
        filter.encoder_digest.as_ref().map(|d| d.as_str()),
        tag_hashes.as_deref(),
        filter.embedding_ids.as_deref(),
        filter.cursor,
        limit as i64
    )
    .fetch_all(tx)
    .await?;

    Ok(similar)
}

#[cfg(test)]
mod tests_search_embeddings {
    use super::super::{
        EmbeddingNew, TagNew,
        queries::{store_embeddings, store_tags},
    };
    use super::{SearchEmbeddingsFilter, search_embeddings};
    use crate::services::indexing::Digest;
    use chrono::{SubsecRound, Utc};
    use uuid::Uuid;

    #[derive(Clone)]
    struct TestData {
        embeddings: Vec<EmbeddingNew>,
        encoder_primary: Digest,
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_cursor(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let results = search_embeddings(
            &pool,
            &SearchEmbeddingsFilter::default().with_cursor(&data.embeddings[0].id),
            10,
        )
        .await?;

        assert_eq!(
            results.iter().map(|r| r.embedding_id).collect::<Vec<_>>(),
            vec![data.embeddings[1].id, data.embeddings[2].id]
        );
        assert!(results.iter().any(|r| {
            r.embedding_id == data.embeddings[2].id && r.tags.contains(&"tail_marker".to_string())
        }));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_multiple_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let results = search_embeddings(
            &pool,
            &SearchEmbeddingsFilter::default()
                .with_tag("multi_one")
                .with_tag("multi_two"),
            10,
        )
        .await?;

        assert_eq!(results.len(), 2);

        let expectations = [
            (data.embeddings[0].id, "multi_one"),
            (data.embeddings[1].id, "multi_two"),
        ];

        for (i, (embedding_id, tag)) in expectations.iter().enumerate() {
            assert_eq!(&results[i].embedding_id, embedding_id);
            assert!(results[i].tags.contains(&tag.to_string()));
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_embedding_ids(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let results = search_embeddings(
            &pool,
            &SearchEmbeddingsFilter::default()
                .with_embedding_id(&data.embeddings[0].id)
                .with_embedding_id(&data.embeddings[2].id),
            10,
        )
        .await?;

        assert_eq!(results.len(), 2);

        let expectations = [
            (
                data.embeddings[0].id,
                ["alpha", "cursor_anchor", "multi_one"],
            ),
            (data.embeddings[2].id, ["combo", "tail_marker", "gamma"]),
        ];

        for (i, (embedding_id, tags)) in expectations.iter().enumerate() {
            assert_eq!(&results[i].embedding_id, embedding_id);
            for tag in tags {
                // Check for each of the expected tags
                assert!(results[i].tags.contains(&tag.to_string()));
            }
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_multiple_options(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let results = search_embeddings(
            &pool,
            &SearchEmbeddingsFilter::default()
                .with_tag("combo")
                .with_encoder_id(&data.encoder_primary)
                .with_cursor(&data.embeddings[0].id),
            5,
        )
        .await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].embedding_id, data.embeddings[1].id);
        assert!(results[0].tags.contains(&"combo".to_string()));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_when_nothing_matches(pool: sqlx::PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let results = search_embeddings(
            &pool,
            &SearchEmbeddingsFilter::default().with_tag("missing"),
            5,
        )
        .await?;

        assert!(results.is_empty());

        Ok(())
    }

    async fn seed(pool: &sqlx::PgPool) -> anyhow::Result<TestData> {
        crate::migrations::run_default_migrations(&pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let encoder_primary =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();
        let encoder_secondary =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000001")
                .unwrap();

        let embeddings = vec![
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000601")?,
                encoder_digest: encoder_primary,
                encoded_at: now,
                value: [0.1; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000602")?,
                encoder_digest: encoder_primary,
                encoded_at: now,
                value: [0.2; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000603")?,
                encoder_digest: encoder_secondary,
                encoded_at: now,
                value: [0.3; super::EMBEDDING_SIZE],
            },
        ];

        let tags = vec![
            TagNew::new("alpha", embeddings[0].id, now),
            TagNew::new("cursor_anchor", embeddings[0].id, now),
            TagNew::new("multi_one", embeddings[0].id, now),
            TagNew::new("combo", embeddings[1].id, now),
            TagNew::new("multi_two", embeddings[1].id, now),
            TagNew::new("beta", embeddings[1].id, now),
            TagNew::new("combo", embeddings[2].id, now),
            TagNew::new("tail_marker", embeddings[2].id, now),
            TagNew::new("gamma", embeddings[2].id, now),
        ];

        store_embeddings(pool, &embeddings).await?;
        store_tags(pool, &tags).await?;

        Ok(TestData {
            embeddings,
            encoder_primary,
        })
    }
}

/// Returns tag pairs that co-occur on the same embedding.
///
/// The query enforces an AND across the two tag sets: an embedding matches
/// only if it has at least one tag from `tags_lhs` and at least one tag from
/// `tags_rhs`. Each returned tuple contains the embedding id plus the matched
/// lhs and rhs tags. Callers can use the pairs to derive their own identifiers
/// (for example by parsing a `genotype_id:<uuid>` tag) while keeping tag
/// semantics outside the repository.
#[instrument(level = "debug", skip(tx, tags_lhs, tags_rhs))]
pub async fn get_tag_pairs_for_embeddings<'tx, E: PgExecutor<'tx>>(
    tx: E,
    tags_lhs: &[String],
    tags_rhs: &[String],
) -> Result<Vec<(Uuid, String, String)>, super::Error> {
    if tags_lhs.is_empty() || tags_rhs.is_empty() {
        return Ok(Vec::new());
    }

    let tags_lhs_hashes = tags_lhs
        .iter()
        .map(|tag| fnv1a_hash_str_32(tag) as i64)
        .collect::<Vec<_>>();

    let tags_rhs_hashes = tags_rhs
        .iter()
        .map(|tag| fnv1a_hash_str_32(tag) as i64)
        .collect::<Vec<_>>();

    let rows = sqlx::query!(
        r#"
        SELECT DISTINCT tg_a.id AS "embedding_id!: Uuid",
                        tg_a.tag_name AS "lhs_tag!",
                        tg_b.tag_name AS "rhs_tag!"
        FROM fx_durable_ga.tagged_embeddings tg_a
        JOIN fx_durable_ga.tagged_embeddings tg_b
            ON tg_b.id = tg_a.id
        WHERE tg_a.tag_hash = ANY($1::BIGINT[])
            AND tg_a.tag_name = ANY($2::TEXT[])
            AND tg_b.tag_hash = ANY($3::BIGINT[])
            AND tg_b.tag_name = ANY($4::TEXT[])
        "#,
        &tags_lhs_hashes,
        &tags_lhs,
        &tags_rhs_hashes,
        &tags_rhs
    )
    .fetch_all(tx)
    .await?;

    let mut pairs = Vec::with_capacity(rows.len());
    for row in rows {
        pairs.push((row.embedding_id, row.lhs_tag, row.rhs_tag));
    }

    Ok(pairs)
}

#[cfg(test)]
mod tests_get_tag_pairs_for_embeddings {
    use super::{EmbeddingNew, TagNew, get_tag_pairs_for_embeddings, store_embeddings, store_tags};
    use crate::services::indexing::Digest;
    use chrono::SubsecRound;
    use chrono::Utc;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    // Returns only pairs where both tags exist on the same embedding.
    async fn it_returns_matching_tag_pairs(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let indexer_id =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000001")
                .unwrap();
        let other_indexer_id =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000002")
                .unwrap();

        let genotype_a = Uuid::now_v7();
        let genotype_b = Uuid::now_v7();

        let embeddings = vec![
            EmbeddingNew {
                id: Uuid::now_v7(),
                encoder_digest: indexer_id,
                encoded_at: now,
                value: [1_f32; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: Uuid::now_v7(),
                encoder_digest: other_indexer_id,
                encoded_at: now,
                value: [2_f32; super::EMBEDDING_SIZE],
            },
        ];

        store_embeddings(&pool, &embeddings).await?;

        let genotype_tag_a = format!("genotype_id:{}", genotype_a);
        let genotype_tag_b = format!("genotype_id:{}", genotype_b);
        let indexer_tag = format!("indexer_id:{}", indexer_id);
        let other_indexer_tag = format!("indexer_id:{}", other_indexer_id);

        let tags = vec![
            TagNew::new(&genotype_tag_a, embeddings[0].id, now),
            TagNew::new(&indexer_tag, embeddings[0].id, now),
            TagNew::new(&genotype_tag_b, embeddings[1].id, now),
            TagNew::new(&other_indexer_tag, embeddings[1].id, now),
        ];

        store_tags(&pool, &tags).await?;

        let lhs_tags = vec![genotype_tag_a.clone(), genotype_tag_b];
        let rhs_tags = vec![indexer_tag.clone()];

        let pairs = get_tag_pairs_for_embeddings(&pool, &lhs_tags, &rhs_tags).await?;

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, embeddings[0].id);
        assert_eq!(pairs[0].1, genotype_tag_a);
        assert_eq!(pairs[0].2, indexer_tag);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    // Returns one row per matching lhs tag on the same embedding.
    async fn it_returns_multiple_lhs_matches(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let indexer_id =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000001")
                .unwrap();

        let genotype_a = Uuid::now_v7();
        let genotype_b = Uuid::now_v7();
        let embedding_id = Uuid::now_v7();

        let embeddings = vec![EmbeddingNew {
            id: embedding_id,
            encoder_digest: indexer_id,
            encoded_at: now,
            value: [1_f32; super::EMBEDDING_SIZE],
        }];

        store_embeddings(&pool, &embeddings).await?;

        let genotype_tag_a = format!("genotype_id:{}", genotype_a);
        let genotype_tag_b = format!("genotype_id:{}", genotype_b);
        let indexer_tag = format!("indexer_id:{}", indexer_id);
        let tags = vec![
            TagNew::new(&genotype_tag_a, embedding_id, now),
            TagNew::new(&genotype_tag_b, embedding_id, now),
            TagNew::new(&indexer_tag, embedding_id, now),
        ];

        store_tags(&pool, &tags).await?;

        let lhs_tags = vec![genotype_tag_a.clone(), genotype_tag_b.clone()];
        let rhs_tags = vec![indexer_tag.clone()];
        let mut pairs = get_tag_pairs_for_embeddings(&pool, &lhs_tags, &rhs_tags).await?;
        pairs.sort_by(|a, b| a.1.cmp(&b.1));

        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, embedding_id);
        assert_eq!(pairs[0].1, genotype_tag_a);
        assert_eq!(pairs[0].2, indexer_tag);
        assert_eq!(pairs[1].0, embedding_id);
        assert_eq!(pairs[1].1, genotype_tag_b);
        assert_eq!(pairs[1].2, indexer_tag);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    // Returns no rows when either input tag set is empty.
    async fn it_returns_empty_for_empty_inputs(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lhs_tags: Vec<String> = Vec::new();
        let rhs_tags: Vec<String> = Vec::new();

        let pairs = get_tag_pairs_for_embeddings(&pool, &lhs_tags, &rhs_tags).await?;
        assert!(pairs.is_empty());

        Ok(())
    }
}

#[derive(Debug)]
pub struct SearchSimilarEmbeddingsFilter {
    reference_tags: Option<Vec<String>>,
    search_tags: Option<Vec<String>>,
    order: Option<SortOrder>,
}

impl SearchSimilarEmbeddingsFilter {
    pub fn with_reference_tag(mut self, tag: &str) -> Self {
        self.reference_tags
            .get_or_insert_with(Vec::new)
            .push(tag.to_string());
        self
    }

    pub fn with_search_tag(mut self, tag: &str) -> Self {
        self.search_tags
            .get_or_insert_with(Vec::new)
            .push(tag.to_string());
        self
    }

    pub fn with_order(mut self, order: SortOrder) -> Self {
        self.order = Some(order);
        self
    }
}

impl Default for SearchSimilarEmbeddingsFilter {
    fn default() -> Self {
        Self {
            reference_tags: None,
            search_tags: None,
            order: Some(SortOrder::Asc),
        }
    }
}

#[instrument(level = "debug", skip(tx))]
/// Search embeddings by distance to a reference centroid.
///
/// The reference centroid is computed from embeddings that match
/// `filter.reference_tags` (OR logic). The search results are limited
/// to embeddings that match `filter.search_tags` (OR logic). Both the
/// centroid and search space are always restricted to `encoder_digest`.
///
/// If either tag set is `None`, that side defaults to all embeddings
/// for the encoder. If a tag set is provided but empty, the query
/// returns no rows.
///
/// Tags use OR logic within this query. For more complex tag filtering,
/// compose with `search_embeddings` to pre-select embeddings before
/// comparing similarity.
pub async fn search_similar_embeddings<'tx, E: PgExecutor<'tx>>(
    tx: E,
    encoder_digest: &Digest,
    filter: &SearchSimilarEmbeddingsFilter,
    limit: i64,
) -> Result<Vec<(Embedding, f64)>, super::Error> {
    if filter
        .reference_tags
        .as_ref()
        .is_some_and(|tags| tags.is_empty())
        || filter
            .search_tags
            .as_ref()
            .is_some_and(|tags| tags.is_empty())
    {
        return Ok(Vec::new());
    }

    let reference_tag_hashes = filter.reference_tags.as_ref().map(|tags| {
        tags.iter()
            .map(|tag| fnv1a_hash_str_32(tag) as i64)
            .collect::<Vec<i64>>()
    });

    let search_tag_hashes = filter.search_tags.as_ref().map(|tags| {
        tags.iter()
            .map(|tag| fnv1a_hash_str_32(tag) as i64)
            .collect::<Vec<i64>>()
    });

    let order = filter.order.as_ref().unwrap_or(&SortOrder::Asc).to_string();

    // Query flow:
    // 1) Build a centroid from embeddings matching reference tags (OR) and encoder.
    // 2) Compare each search-space embedding (matching search tags OR) to that centroid.
    // 3) Return embeddings with aggregated tags, ordered by distance.
    let rows = sqlx::query!(
        r#"
        WITH reference_centroid AS (
            SELECT AVG(e.value) AS centroid
            FROM fx_durable_ga.embeddings e
            WHERE e.encoder_digest = $1::CHAR(64)
                AND (
                    $2::BIGINT[] IS NULL
                    OR e.id IN (
                        SELECT et.embedding_id
                        FROM fx_durable_ga.embedding_tags et
                        WHERE et.tag_hash = ANY($2::BIGINT[])
                    )
                )
        )
        SELECT
            e.id AS "embedding_id!: Uuid",
            e.value <=> rc.centroid AS "distance!: f64",
            ARRAY_AGG(t.tag_name ORDER BY t.tag_name ASC) AS "tags!: Vec<String>"
        FROM fx_durable_ga.embeddings e
        CROSS JOIN reference_centroid rc
        LEFT JOIN fx_durable_ga.embedding_tags t ON e.id = t.embedding_id
        WHERE
            rc.centroid IS NOT NULL
            AND e.encoder_digest = $1::CHAR(64)
            AND (
                $3::BIGINT[] IS NULL
                OR e.id IN (
                    SELECT et.embedding_id
                    FROM fx_durable_ga.embedding_tags et
                    WHERE et.tag_hash = ANY($3::BIGINT[])
                )
            )
        GROUP BY e.id, e.value, rc.centroid
        ORDER BY
            CASE WHEN $4::TEXT = 'asc' THEN e.value <=> rc.centroid END ASC,
            CASE WHEN $4::TEXT = 'desc' THEN e.value <=> rc.centroid END DESC,
            e.id
        LIMIT $5;
        "#,
        encoder_digest.as_str(),
        reference_tag_hashes.as_deref(),
        search_tag_hashes.as_deref(),
        order,
        limit
    )
    .fetch_all(tx)
    .await?;

    let tagged_embeddings_with_distance = rows
        .into_iter()
        .map(|row| {
            let tagged_embedding = Embedding {
                embedding_id: row.embedding_id,
                tags: row.tags,
            };
            (tagged_embedding, row.distance)
        })
        .collect();

    Ok(tagged_embeddings_with_distance)
}

#[cfg(test)]
mod tests_search_similar_embeddings {
    use super::super::{EmbeddingNew, TagNew};
    use super::{
        SearchSimilarEmbeddingsFilter, search_similar_embeddings, store_embeddings, store_tags,
    };
    use crate::repositories::ordering::SortOrder;
    use crate::services::indexing::Digest;
    use chrono::SubsecRound;
    use chrono::Utc;
    use uuid::Uuid;

    struct TestData {
        encoder_primary: Digest,
        ref_id: Uuid,
        near_id: Uuid,
        mid_id: Uuid,
        far_id: Uuid,
        extra_id: Uuid,
        secondary_id: Uuid,
    }

    #[sqlx::test(migrations = false)]
    /// Uses default filters (None) to compare the entire encoder population
    /// against its centroid, returning only primary-encoder embeddings.
    async fn it_compares_entire_population_by_default(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let similar = search_similar_embeddings(
            &pool,
            &data.encoder_primary,
            &SearchSimilarEmbeddingsFilter::default(),
            10,
        )
        .await?;

        let mut ids = similar
            .iter()
            .map(|(e, _)| e.embedding_id)
            .collect::<Vec<_>>();
        ids.sort();

        let mut expected = vec![
            data.ref_id,
            data.near_id,
            data.mid_id,
            data.far_id,
            data.extra_id,
        ];
        expected.sort();

        assert_eq!(ids, expected);
        assert!(!ids.contains(&data.secondary_id));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    /// Filters by reference and search tags to validate OR tag matching
    /// and that results are limited to the search tag set.
    async fn it_filters_reference_and_search_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let filter = SearchSimilarEmbeddingsFilter::default()
            .with_reference_tag("ref_a")
            .with_search_tag("search_a");

        let similar = search_similar_embeddings(&pool, &data.encoder_primary, &filter, 10).await?;

        assert_eq!(similar.len(), 2);
        assert_eq!(similar[0].0.embedding_id, data.ref_id);
        assert_eq!(similar[1].0.embedding_id, data.near_id);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    /// Ensures multiple search tags are treated as OR, returning embeddings
    /// that match any of the provided search tags.
    async fn it_uses_or_logic_for_search_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let filter = SearchSimilarEmbeddingsFilter::default()
            .with_reference_tag("ref_a")
            .with_search_tag("search_a")
            .with_search_tag("search_b");

        let similar = search_similar_embeddings(&pool, &data.encoder_primary, &filter, 10).await?;

        let mut ids = similar
            .iter()
            .map(|(e, _)| e.embedding_id)
            .collect::<Vec<_>>();
        ids.sort();

        let mut expected = vec![
            data.ref_id,
            data.near_id,
            data.mid_id,
            data.far_id,
            data.extra_id,
        ];
        expected.sort();

        assert_eq!(ids, expected);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    /// Confirms descending order returns the furthest embedding first,
    /// which is useful for outlier detection.
    async fn it_orders_desc_for_outliers(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let filter = SearchSimilarEmbeddingsFilter::default()
            .with_reference_tag("ref_a")
            .with_search_tag("search_a")
            .with_order(SortOrder::Desc);

        let similar = search_similar_embeddings(&pool, &data.encoder_primary, &filter, 10).await?;

        assert_eq!(similar.len(), 2);
        assert_eq!(similar[0].0.embedding_id, data.near_id);
        assert_eq!(similar[1].0.embedding_id, data.ref_id);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    /// Returns empty results when the search tags match no embeddings,
    /// even if the reference tags match.
    async fn it_returns_empty_for_missing_search_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let filter = SearchSimilarEmbeddingsFilter::default()
            .with_reference_tag("ref_a")
            .with_search_tag("missing");

        let similar = search_similar_embeddings(&pool, &data.encoder_primary, &filter, 10).await?;

        assert!(similar.is_empty());

        Ok(())
    }

    async fn seed(pool: &sqlx::PgPool) -> anyhow::Result<TestData> {
        crate::migrations::run_default_migrations(pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let encoder_primary =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();
        let encoder_secondary =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000001")
                .unwrap();

        let build_vector = |x: f32, y: f32| {
            let mut value = [0.0_f32; super::EMBEDDING_SIZE];
            value[0] = x;
            value[1] = y;
            value
        };

        let embeddings = vec![
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000200")?,
                encoder_digest: encoder_primary,
                encoded_at: now,
                value: build_vector(1.0, 0.0),
            },
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000201")?,
                encoder_digest: encoder_primary,
                encoded_at: now,
                value: build_vector(1.0, 0.1),
            },
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000202")?,
                encoder_digest: encoder_primary,
                encoded_at: now,
                value: build_vector(0.8, 0.4),
            },
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000203")?,
                encoder_digest: encoder_primary,
                encoded_at: now,
                value: build_vector(0.0, 1.0),
            },
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000204")?,
                encoder_digest: encoder_primary,
                encoded_at: now,
                value: build_vector(0.6, 0.2),
            },
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000205")?,
                encoder_digest: encoder_secondary,
                encoded_at: now,
                value: build_vector(1.0, 0.0),
            },
        ];

        let tags = vec![
            TagNew::new("ref_a", embeddings[0].id, now),
            TagNew::new("search_a", embeddings[0].id, now),
            TagNew::new("search_a", embeddings[1].id, now),
            TagNew::new("search_b", embeddings[2].id, now),
            TagNew::new("ref_b", embeddings[3].id, now),
            TagNew::new("search_b", embeddings[3].id, now),
            TagNew::new("search_b", embeddings[4].id, now),
            TagNew::new("search_a", embeddings[5].id, now),
        ];

        store_embeddings(&*pool, embeddings.iter()).await?;
        store_tags(&*pool, tags.iter()).await?;

        Ok(TestData {
            encoder_primary,
            ref_id: embeddings[0].id,
            near_id: embeddings[1].id,
            mid_id: embeddings[2].id,
            far_id: embeddings[3].id,
            extra_id: embeddings[4].id,
            secondary_id: embeddings[5].id,
        })
    }
}

#[instrument(level = "debug", skip(tx, embeddings))]
pub(crate) async fn store_embeddings<'tx, 'a, I, E>(
    tx: E,
    embeddings: I,
) -> Result<Vec<Uuid>, super::Error>
where
    I: IntoIterator<Item = &'a EmbeddingNew>,
    E: PgExecutor<'tx>,
{
    let mut embeddings = embeddings.into_iter().peekable();
    if embeddings.peek().is_none() {
        return Ok(vec![]);
    }

    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.embeddings (
            id,
            encoded_at,
            encoder_digest,
            value
        )
        VALUES ",
    );

    let mut first = true;
    for e in embeddings {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        let value = Vector::from(e.value.to_vec());
        query_builder
            .push("(")
            .push_bind(e.id)
            .push(", ")
            .push_bind(e.encoded_at)
            .push(", ")
            .push_bind(e.encoder_digest.as_str())
            .push(", ")
            .push_bind(value)
            .push(")");
    }

    query_builder.push(" RETURNING id, encoded_at, encoder_digest, value;");

    let query = query_builder.build();
    let rows = query.fetch_all(tx).await?;

    rows.into_iter()
        .map(|row| Ok(row.get::<Uuid, _>("id")))
        .collect()
}

#[cfg(test)]
mod tests_store_embeddings {
    use super::super::{EmbeddingNew, queries::store_embeddings};
    use crate::services::indexing::Digest;
    use chrono::SubsecRound;
    use chrono::Utc;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_stores_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let encoder_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();

        let embeddings = &[EmbeddingNew {
            id: Uuid::nil(),
            encoder_digest,
            encoded_at: Utc::now().trunc_subsecs(6),
            value: [1_f32; super::EMBEDDING_SIZE],
        }];

        let stored = store_embeddings(&pool, embeddings).await?;

        for (e, id) in embeddings.iter().zip(stored) {
            assert_eq!(e.id, id);
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_vec_for_empty_input(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let stored = store_embeddings(&pool, &[]).await?;

        assert!(stored.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_inserts_multiple_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let encoder_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();

        let embeddings = &[
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000020").unwrap(),
                encoder_digest: encoder_digest,
                encoded_at: now,
                value: [2_f32; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000021").unwrap(),
                encoder_digest: encoder_digest,
                encoded_at: now,
                value: [3_f32; super::EMBEDDING_SIZE],
            },
        ];

        let stored = store_embeddings(&pool, embeddings).await?;

        assert_eq!(stored.len(), embeddings.len());

        for (e, id) in embeddings.iter().zip(stored.iter()) {
            assert_eq!(&e.id, id);
        }

        Ok(())
    }
}

#[instrument(level = "debug", skip(tx, tags))]
pub(crate) async fn store_tags<'tx, 'a, I, E>(
    tx: E,
    tags: I,
) -> Result<Vec<(Uuid, String)>, super::Error>
where
    I: IntoIterator<Item = &'a TagNew>,
    E: PgExecutor<'tx>,
{
    let mut tags = tags.into_iter().peekable();
    if tags.peek().is_none() {
        return Ok(vec![]);
    }

    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.embedding_tags (
            id,
            tag_hash,
            tag_name,
            embedding_id,
            tagged_at
        )
        VALUES ",
    );

    let mut first = true;
    for t in tags {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        let tag_hash = fnv1a_hash_str_32(&t.tag_name) as i64;

        query_builder
            .push("(")
            .push_bind(t.id)
            .push(", ")
            .push_bind(tag_hash)
            .push(", ")
            .push_bind(&t.tag_name)
            .push(", ")
            .push_bind(t.embedding_id)
            .push(", ")
            .push_bind(t.tagged_at)
            .push(")");
    }

    query_builder.push(" RETURNING embedding_id, tag_name;");

    query_builder
        .build()
        .fetch_all(tx)
        .await?
        .into_iter()
        .map(|row| {
            Ok((
                row.get::<Uuid, _>("embedding_id"),
                row.get::<String, _>("tag_name"),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests_store_tags {
    use super::super::{
        EmbeddingNew, SearchEmbeddingsFilter, TagNew,
        queries::{search_embeddings, store_embeddings, store_tags},
    };
    use crate::services::indexing::Digest;
    use chrono::SubsecRound;
    use chrono::Utc;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_stores_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let encoder_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();

        let embeddings = &[EmbeddingNew {
            id: Uuid::nil(),
            encoder_digest,
            encoded_at: now,
            value: [1_f32; super::EMBEDDING_SIZE],
        }];

        let tags = &[
            TagNew::new("tag_1", embeddings[0].id, now),
            TagNew::new("tag_2", embeddings[0].id, now),
            TagNew::new("tag_3", embeddings[0].id, now),
        ];

        store_embeddings(&pool, embeddings).await?;
        let stored = store_tags(&pool, tags).await?;

        assert!(stored.len() == tags.len());

        let found = search_embeddings(
            &pool,
            &SearchEmbeddingsFilter::default().with_embedding_id(&embeddings[0].id),
            10,
        )
        .await?;

        let expectations = [(embeddings[0].id, ["tag_1", "tag_2", "tag_3"])];

        for (i, (embedding_id, tags)) in expectations.iter().enumerate() {
            assert_eq!(&found[i].embedding_id, embedding_id);
            for tag in tags {
                assert!(found[i].tags.contains(&tag.to_string()));
            }
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_vec_for_empty_input(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let stored = store_tags(&pool, &[]).await?;

        assert_eq!(Vec::<(Uuid, String)>::new(), stored);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_error_on_duplicate_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let encoder_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();

        let embeddings = &[EmbeddingNew {
            id: Uuid::nil(),
            encoder_digest,
            encoded_at: now,
            value: [1_f32; super::EMBEDDING_SIZE],
        }];

        let tags = &[
            TagNew::new("duplicate", embeddings[0].id, now),
            TagNew::new("duplicate", embeddings[0].id, now),
        ];

        store_embeddings(&pool, embeddings).await?;
        let result = store_tags(&pool, tags).await;

        assert!(result.is_err());

        Ok(())
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub struct RequestedEmbedding {
    pub(crate) entity_id: Uuid,
    pub(crate) entity_type_name: String,
    pub(crate) indexer_id: Digest,
    pub(crate) metadata: serde_json::Value,
    requested_at: DateTime<Utc>,
}

impl RequestedEmbedding {
    pub(crate) fn new(
        entity_id: Uuid,
        entity_type_name: String,
        indexer_id: Digest,
        metadata: serde_json::Value,
        requested_at: DateTime<Utc>,
    ) -> Self {
        Self {
            entity_id,
            entity_type_name,
            indexer_id,
            metadata,
            requested_at,
        }
    }

    pub(crate) fn next_cursor(&self) -> (DateTime<Utc>, Uuid) {
        (self.requested_at, self.entity_id)
    }

    #[cfg(test)]
    fn id(&self) -> (Uuid, Digest) {
        (self.entity_id, self.indexer_id)
    }
}

/// Store records of requested embeddings that could not yet
/// be computed. requested_embeddings is an outbox table of
/// deferred indexation work.
#[instrument(level = "debug", skip(tx, requested_embeddings))]
pub(crate) async fn store_requested_embeddings<'tx, 'a, I, E>(
    tx: E,
    requested_embeddings: I,
) -> Result<Vec<RequestedEmbedding>, super::Error>
where
    I: IntoIterator<Item = &'a RequestedEmbedding>,
    E: PgExecutor<'tx>,
{
    let mut requested_embeddings = requested_embeddings.into_iter().peekable();
    if requested_embeddings.peek().is_none() {
        return Ok(vec![]);
    }

    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.requested_embeddings (
            entity_id,
            entity_type_name,
            indexer_digest,
            metadata,
            requested_at
        )
        VALUES ",
    );

    let mut first = true;
    for re in requested_embeddings {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        query_builder
            .push("(")
            .push_bind(re.entity_id)
            .push(", ")
            .push_bind(&re.entity_type_name)
            .push(", ")
            .push_bind(re.indexer_id)
            .push(", ")
            .push_bind(&re.metadata)
            .push(", ")
            .push_bind(re.requested_at)
            .push(")");
    }

    query_builder.push(
        " RETURNING entity_id, entity_type_name, indexer_digest::TEXT, metadata, requested_at;",
    );

    query_builder
        .build()
        .fetch_all(tx)
        .await?
        .into_iter()
        .map(|row| {
            Ok(RequestedEmbedding {
                entity_id: row.get::<Uuid, _>("entity_id"),
                entity_type_name: row.get::<String, _>("entity_type_name"),
                indexer_id: row.get::<Digest, _>("indexer_digest"),
                metadata: row.get::<serde_json::Value, _>("metadata"),
                requested_at: row.get::<DateTime<Utc>, _>("requested_at"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests_store_requested_embeddings {
    use super::{RequestedEmbedding, store_requested_embeddings};
    use crate::services::indexing::Digest;
    use chrono::{SubsecRound, Utc};
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_stores_requested_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let requested_at = Utc::now().trunc_subsecs(6);

        let requests = vec![
            RequestedEmbedding {
                entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000a101")?,
                entity_type_name: "genotype".to_string(),
                indexer_id: Digest::from_hex(
                    "0000000000000000000000000000000000000000000000000000000000000010",
                )?,
                metadata: serde_json::Value::Object(Default::default()),
                requested_at,
            },
            RequestedEmbedding {
                entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000a102")?,
                entity_type_name: "morphology".to_string(),
                indexer_id: Digest::from_hex(
                    "0000000000000000000000000000000000000000000000000000000000000011",
                )?,
                metadata: serde_json::Value::Object(Default::default()),
                requested_at,
            },
        ];

        let mut stored = store_requested_embeddings(&pool, &requests).await?;

        assert_eq!(stored.len(), requests.len());

        let mut expected = requests.clone();
        expected.sort_by_key(|r| r.entity_id);
        stored.sort_by_key(|r| r.entity_id);

        for (expected, actual) in expected.iter().zip(stored.iter()) {
            assert_eq!(actual.entity_id, expected.entity_id);
            assert_eq!(actual.entity_type_name, expected.entity_type_name);
            assert_eq!(actual.indexer_id, expected.indexer_id);
            assert_eq!(actual.requested_at, requested_at);
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_conflicts(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let requested_at = Utc::now().trunc_subsecs(6);

        let conflict = RequestedEmbedding {
            entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000a000")?,
            entity_type_name: "entity_type_name".to_string(),
            indexer_id: Digest::from_hex(
                "0000000000000000000000000000000000000000000000000000000000000010",
            )?,
            metadata: serde_json::Value::Object(Default::default()),
            requested_at,
        };

        store_requested_embeddings(&pool, [&conflict]).await?;

        let result = store_requested_embeddings(&pool, [&conflict]).await;

        assert!(result.is_err());

        Ok(())
    }
}

#[derive(Debug)]
pub struct SearchRequestedEmbeddingsFilter {
    entity_type_name: Option<String>,
    indexer_id: Option<Digest>,
    cursor: Option<(DateTime<Utc>, Uuid)>,
    is_handled: bool,
}

impl Default for SearchRequestedEmbeddingsFilter {
    fn default() -> Self {
        Self {
            entity_type_name: None,
            indexer_id: None,
            cursor: None,
            is_handled: false,
        }
    }
}

impl SearchRequestedEmbeddingsFilter {
    pub fn with_indexer_id(mut self, indexer_id: &Digest) -> Self {
        self.indexer_id = Some(*indexer_id);
        self
    }

    pub fn with_entity_type_name(mut self, entity_type_name: &str) -> Self {
        self.entity_type_name = Some(entity_type_name.to_string());
        self
    }

    pub fn with_cursor(mut self, cursor: &(DateTime<Utc>, Uuid)) -> Self {
        self.cursor = Some(*cursor);
        self
    }

    pub fn with_is_handled(mut self) -> Self {
        self.is_handled = true;
        self
    }
}

// Searches and paginates over RequestedEmbeddings
#[instrument(level = "debug", skip(tx))]
pub async fn search_requested_embeddings<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &SearchRequestedEmbeddingsFilter,
    limit: i64,
) -> Result<Vec<RequestedEmbedding>, super::Error> {
    let rows = sqlx::query!(
        r#"
        SELECT
            entity_id,
            entity_type_name,
            indexer_digest::TEXT AS "indexer_id: String",
            metadata,
            requested_at
        FROM fx_durable_ga.requested_embeddings
        WHERE
            (handled_at IS NULL OR $1::BOOL)
            AND (
                $2::CHAR(64) IS NULL
                OR indexer_digest = $2::CHAR(64)
            )
            AND (
                $3::TEXT IS NULL
                OR entity_type_name = $3::TEXT
            )
            AND (
                $4::TIMESTAMPTZ IS NULL
                OR requested_at > $4::TIMESTAMPTZ
                OR (requested_at = $4::TIMESTAMPTZ AND entity_id > $5::UUID)
            )
        ORDER BY requested_at, entity_id
        LIMIT $6;
        "#,
        filter.is_handled,
        filter.indexer_id.as_ref().map(|d| d.as_str()),
        filter.entity_type_name,
        filter.cursor.as_ref().map(|c| c.0), // requested_at
        filter.cursor.as_ref().map(|c| c.1), // entity id
        limit as i64
    )
    .fetch_all(tx)
    .await?;

    rows.into_iter()
        .map(|row| {
            Ok(RequestedEmbedding {
                entity_id: row.entity_id,
                entity_type_name: row.entity_type_name,
                indexer_id: Digest::from_hex(&row.indexer_id.unwrap_or_default())
                    .map_err(|e| super::Error::Tx(anyhow::Error::new(e)))?,
                metadata: row.metadata,
                requested_at: row.requested_at,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests_search_requested_embeddings {
    use super::{
        RequestedEmbedding, SearchRequestedEmbeddingsFilter, search_requested_embeddings,
        store_requested_embeddings,
    };
    use crate::services::indexing::Digest;
    use chrono::{Duration, SubsecRound, Utc};
    use uuid::Uuid;

    struct TestData {
        pool: sqlx::PgPool,
        records: Vec<RequestedEmbedding>,
        primary_digest: Digest,
        _secondary_digest: Digest,
        primary_entity_type_name: String,
    }

    #[sqlx::test(migrations = false)]
    async fn it_searches_requested_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let filter = SearchRequestedEmbeddingsFilter::default();

        let mut results = search_requested_embeddings(&data.pool, &filter, 10).await?;
        let mut expected = data.records.clone();

        assert_eq!(results.len(), expected.len());

        results.sort_by_key(|r| (r.requested_at, r.entity_id, r.indexer_id));
        expected.sort_by_key(|r| (r.requested_at, r.entity_id, r.indexer_id));

        for (actual, expected) in results.iter().zip(expected.iter()) {
            assert_eq!(actual.entity_id, expected.entity_id);
            assert_eq!(actual.indexer_id, expected.indexer_id);
            assert_eq!(actual.entity_type_name, expected.entity_type_name);
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_paginates_requested_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let page_one_filter = SearchRequestedEmbeddingsFilter::default();

        let first_page = search_requested_embeddings(&data.pool, &page_one_filter, 2).await?;

        assert_eq!(first_page.len(), 2);

        let cursor = first_page
            .last()
            .expect("Expected at least one result")
            .next_cursor();

        let page_two_filter = SearchRequestedEmbeddingsFilter::default().with_cursor(&cursor);

        let mut second_page = search_requested_embeddings(&data.pool, &page_two_filter, 10).await?;

        let mut expected: Vec<_> = data
            .records
            .iter()
            .skip(first_page.len())
            .cloned()
            .collect();

        second_page.sort_by_key(|r| (r.requested_at, r.entity_id, r.indexer_id));
        expected.sort_by_key(|r| (r.requested_at, r.entity_id, r.indexer_id));

        assert_eq!(second_page.len(), expected.len());
        for (actual, expected) in second_page.iter().zip(expected.iter()) {
            assert_eq!(actual.entity_id, expected.entity_id);
            assert_eq!(actual.indexer_id, expected.indexer_id);
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_optionally_filters_for_indexer_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let filter =
            SearchRequestedEmbeddingsFilter::default().with_indexer_id(&data.primary_digest);

        let mut results = search_requested_embeddings(&data.pool, &filter, 10).await?;

        let mut expected: Vec<_> = data
            .records
            .iter()
            .filter(|r| r.indexer_id == data.primary_digest)
            .cloned()
            .collect();

        results.sort_by_key(|r| (r.requested_at, r.entity_id, r.indexer_id));
        expected.sort_by_key(|r| (r.requested_at, r.entity_id, r.indexer_id));

        assert_eq!(results.len(), expected.len());
        for (actual, expected) in results.iter().zip(expected.iter()) {
            assert_eq!(actual.entity_id, expected.entity_id);
            assert_eq!(actual.indexer_id, expected.indexer_id);
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_optionally_filters_for_entity_type_name(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let filter = SearchRequestedEmbeddingsFilter::default()
            .with_entity_type_name(&data.primary_entity_type_name);

        let mut results = search_requested_embeddings(&data.pool, &filter, 10).await?;

        let mut expected: Vec<_> = data
            .records
            .iter()
            .filter(|r| r.entity_type_name == data.primary_entity_type_name)
            .cloned()
            .collect();

        results.sort_by_key(|r| (r.requested_at, r.entity_id, r.indexer_id));
        expected.sort_by_key(|r| (r.requested_at, r.entity_id, r.indexer_id));

        assert_eq!(results.len(), expected.len());
        for (actual, expected) in results.iter().zip(expected.iter()) {
            assert_eq!(actual.entity_id, expected.entity_id);
            assert_eq!(actual.indexer_id, expected.indexer_id);
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_vec_for_no_match(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        seed(&pool).await?;

        let missing_digest =
            Digest::from_hex("00000000000000000000000000000000000000000000000000000000000000ff")?;

        let filter = SearchRequestedEmbeddingsFilter::default().with_indexer_id(&missing_digest);

        let results = search_requested_embeddings(&pool, &filter, 5).await?;

        assert!(results.is_empty());

        Ok(())
    }

    async fn seed(pool: &sqlx::PgPool) -> anyhow::Result<TestData> {
        let base_time = Utc::now().trunc_subsecs(6);
        let primary_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000020")?;
        let secondary_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000021")?;

        let primary_entity_type_name = "genotype".to_string();

        let requests = vec![
            RequestedEmbedding {
                entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000b001")?,
                entity_type_name: primary_entity_type_name.clone(),
                indexer_id: primary_digest,
                metadata: serde_json::Value::Object(Default::default()),
                requested_at: base_time - Duration::seconds(3),
            },
            RequestedEmbedding {
                entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000b002")?,
                entity_type_name: primary_entity_type_name.clone(),
                indexer_id: primary_digest,
                metadata: serde_json::Value::Object(Default::default()),
                requested_at: base_time - Duration::seconds(2),
            },
            RequestedEmbedding {
                entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000b003")?,
                entity_type_name: "morphology".to_string(),
                indexer_id: secondary_digest,
                metadata: serde_json::Value::Object(Default::default()),
                requested_at: base_time - Duration::seconds(1),
            },
            RequestedEmbedding {
                entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000b004")?,
                entity_type_name: "phenotype".to_string(),
                indexer_id: primary_digest,
                metadata: serde_json::Value::Object(Default::default()),
                requested_at: base_time,
            },
        ];

        let records = store_requested_embeddings(pool, &requests).await?;

        Ok(TestData {
            pool: pool.clone(),
            records,
            primary_digest,
            _secondary_digest: secondary_digest,
            primary_entity_type_name,
        })
    }
}

// Sets handled_at for the specified RequestedEmbeddings
#[instrument(level = "debug", skip(tx, requested_embedding_ids))]
pub(crate) async fn set_requested_embeddings_handled_at<'tx, 'a, I, E>(
    tx: E,
    requested_embedding_ids: I,
    handled_at: DateTime<Utc>,
) -> Result<u64, super::Error>
where
    I: IntoIterator<Item = &'a (Uuid, Digest)>,
    E: PgExecutor<'tx>,
{
    let mut requested_embedding_ids = requested_embedding_ids.into_iter().peekable();
    if requested_embedding_ids.peek().is_none() {
        return Ok(0);
    }

    let (uuids, digests): (Vec<Uuid>, Vec<String>) = requested_embedding_ids
        .map(|id| (id.0, id.1.as_str().to_string()))
        .unzip();

    let result = sqlx::query!(
        r#"
        UPDATE fx_durable_ga.requested_embeddings re
        SET handled_at = $1
        FROM UNNEST($2::UUID[], $3::CHAR(64)[]) AS ids(entity_id, indexer_digest)
        WHERE re.handled_at IS NULL
          AND re.entity_id = ids.entity_id
          AND re.indexer_digest = ids.indexer_digest;
        "#,
        handled_at,
        &uuids as &[Uuid],
        &digests as &[String],
    )
    .execute(tx)
    .await?;

    Ok(result.rows_affected())
}

#[cfg(test)]
mod tests_set_requested_embeddings_handled_at {
    use super::{
        RequestedEmbedding, SearchRequestedEmbeddingsFilter, search_requested_embeddings,
        set_requested_embeddings_handled_at, store_requested_embeddings,
    };
    use crate::services::indexing::Digest;
    use chrono::{Duration, SubsecRound, Utc};
    use uuid::Uuid;

    struct TestData {
        pool: sqlx::PgPool,
        requested: Vec<RequestedEmbedding>,
    }

    #[sqlx::test(migrations = false)]
    async fn it_sets_requested_embeddings_to_handled_at(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let target_ids: Vec<(Uuid, Digest)> =
            data.requested.iter().take(2).map(|r| r.id()).collect();

        let handled_at = Utc::now().trunc_subsecs(6);

        let updated =
            set_requested_embeddings_handled_at(&data.pool, target_ids.iter(), handled_at).await?;

        assert_eq!(updated, target_ids.len() as u64);

        let remaining = search_requested_embeddings(
            &data.pool,
            &SearchRequestedEmbeddingsFilter::default(),
            10,
        )
        .await?;

        assert_eq!(remaining.len(), data.requested.len() - target_ids.len());

        let remaining_ids: Vec<_> = remaining.iter().map(|r| r.id()).collect();

        for id in target_ids {
            assert!(!remaining_ids.contains(&id));
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_ignores_unknown_records(pool: sqlx::PgPool) -> anyhow::Result<()> {
        const HANDLE_COUNT: usize = 2;

        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let mut target_ids: Vec<(Uuid, Digest)> = data
            .requested
            .iter()
            .take(HANDLE_COUNT)
            .map(|r| r.id())
            .collect();

        let unknown_id = (
            Uuid::now_v7(),
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000123")
                .unwrap(),
        );

        target_ids.push(unknown_id);

        let handled_at = Utc::now().trunc_subsecs(6);

        let updated =
            set_requested_embeddings_handled_at(&data.pool, target_ids.iter(), handled_at).await?;

        assert_eq!(updated, HANDLE_COUNT as u64);

        let remaining = search_requested_embeddings(
            &data.pool,
            &SearchRequestedEmbeddingsFilter::default(),
            10,
        )
        .await?;

        assert_eq!(remaining.len(), data.requested.len() - HANDLE_COUNT);

        let remaining_ids: Vec<_> = remaining.iter().map(|r| r.id()).collect();

        for id in target_ids {
            assert!(!remaining_ids.contains(&id));
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_is_idempotent(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        let ids: Vec<(Uuid, Digest)> = data.requested.iter().map(|r| r.id()).collect();
        let first_handled_at = Utc::now().trunc_subsecs(6);

        let first =
            set_requested_embeddings_handled_at(&data.pool, ids.iter(), first_handled_at).await?;

        assert_eq!(first, ids.len() as u64);

        let second_handled_at = first_handled_at + Duration::seconds(5);

        let second =
            set_requested_embeddings_handled_at(&data.pool, ids.iter(), second_handled_at).await?;

        assert_eq!(second, 0);

        let filter = SearchRequestedEmbeddingsFilter::default();
        let remaining = search_requested_embeddings(&data.pool, &filter, 10).await?;
        assert!(remaining.is_empty());

        let remaining_after_retry = search_requested_embeddings(&data.pool, &filter, 10).await?;
        assert!(remaining_after_retry.is_empty());

        Ok(())
    }

    async fn seed(pool: &sqlx::PgPool) -> anyhow::Result<TestData> {
        let base_time = Utc::now().trunc_subsecs(6);

        let requests = vec![
            RequestedEmbedding {
                entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000c001")?,
                entity_type_name: "genotype".to_string(),
                indexer_id: Digest::from_hex(
                    "0000000000000000000000000000000000000000000000000000000000000030",
                )?,
                metadata: serde_json::Value::Object(Default::default()),
                requested_at: base_time,
            },
            RequestedEmbedding {
                entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000c002")?,
                entity_type_name: "morphology".to_string(),
                indexer_id: Digest::from_hex(
                    "0000000000000000000000000000000000000000000000000000000000000031",
                )?,
                metadata: serde_json::Value::Object(Default::default()),
                requested_at: base_time + Duration::seconds(1),
            },
            RequestedEmbedding {
                entity_id: Uuid::parse_str("00000000-0000-0000-0000-00000000c003")?,
                entity_type_name: "phenotype".to_string(),
                indexer_id: Digest::from_hex(
                    "0000000000000000000000000000000000000000000000000000000000000032",
                )?,
                metadata: serde_json::Value::Object(Default::default()),
                requested_at: base_time + Duration::seconds(2),
            },
        ];

        let requested = store_requested_embeddings(pool, &requests).await?;

        Ok(TestData {
            pool: pool.clone(),
            requested,
        })
    }
}

/// Returns distinct embedding IDs grouped by tag groups with AND logic.
///
/// For each tag group, returns embeddings that have ALL tags in that group.
/// Tags within a group use AND logic: an embedding must have every tag to match.
/// Order of results corresponds to input tag_groups array.
///
/// # Behavior
/// - **Duplicate tags within groups**: If the same tag appears multiple times in a group,
///   the embedding must have that tag multiple times. Callers should deduplicate
///   tag groups if this is not desired behavior.
/// - **Missing tags**: If a tag doesn't exist in the database, no embeddings will match
///   that group (no error, just empty results).
/// - **Empty groups**: Groups with no matching embeddings will have no entries in results.
/// - **Caller scope control**: No encoder_digest filter applied - caller manages
///   embedding scope by including appropriate tags (e.g., indexer_id, generation_id).
///
/// # Returns
/// Vector of (bin_index, embedding_id) pairs, where bin_index corresponds to
/// the input tag_groups array index. Embeddings may appear multiple times if
/// they match multiple groups.
///
/// # Performance
/// - Uses existing indexes on embedding_tags.tag_hash
/// - Memory efficient - no large intermediate arrays
/// - Scales well with varying tag group sizes
#[instrument(level = "debug", skip(tx, tag_groups), fields(tag_groups_count = tag_groups.len()))]
pub async fn get_embeddings_by_tag_groups<'tx, E: PgExecutor<'tx>>(
    tx: E,
    tag_groups: &[&[&str]],
) -> Result<Vec<(i64, Uuid)>, super::Error> {
    let (bin_indices, tag_hashes): (Vec<i64>, Vec<i64>) = tag_groups
        .iter()
        .enumerate()
        .flat_map(|(bin_index, group)| {
            group
                .iter()
                .map(move |tag| (bin_index as i64, fnv1a_hash_str_32(tag) as i64))
        })
        .unzip();

    let rows = sqlx::query!(
        r#"
        WITH bin_definitions AS (
            SELECT bin_index, tag_hash
            FROM UNNEST($1::BIGINT[], $2::BIGINT[]) AS t(bin_index, tag_hash)
        ),
        embedding_matches AS (
            SELECT
                bd.bin_index,
                et.embedding_id,
                COUNT(DISTINCT et.tag_hash) AS matched_tags
            FROM bin_definitions bd
            JOIN embedding_tags et ON et.tag_hash = bd.tag_hash
            GROUP BY bd.bin_index, et.embedding_id
        ),
        required_counts AS (
            SELECT
                bin_index,
                COUNT(DISTINCT tag_hash) AS required_tags
            FROM bin_definitions
            GROUP BY bin_index
        )
        SELECT
            em.bin_index "bin_index!",
            em.embedding_id "embedding_id!: Uuid"
        FROM embedding_matches em
        JOIN required_counts rc ON em.bin_index = rc.bin_index
        WHERE em.matched_tags = rc.required_tags
        ORDER BY em.bin_index, em.embedding_id
        "#,
        bin_indices.as_slice(),
        tag_hashes.as_slice()
    )
    .fetch_all(tx)
    .await?;

    Ok(rows
        .into_iter()
        .map(|row| (row.bin_index, row.embedding_id))
        .collect())
}

#[cfg(test)]
mod tests_get_embeddings_by_tag_groups {
    use super::{EmbeddingNew, TagNew, get_embeddings_by_tag_groups, store_embeddings, store_tags};
    use crate::services::indexing::Digest;
    use chrono::SubsecRound;
    use chrono::Utc;
    use uuid::Uuid;

    /// Tests that an empty tag groups input returns no results.
    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_for_no_tag_groups(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let results = get_embeddings_by_tag_groups(&pool, &[]).await?;
        assert!(results.is_empty());
        Ok(())
    }

    /// Tests that a single tag matches only embeddings with that tag.
    #[sqlx::test(migrations = false)]
    async fn it_finds_embeddings_by_single_tag(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let test_data = seed_test_data(&pool).await?;

        let results = get_embeddings_by_tag_groups(&pool, &[&["single_tag"]]).await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[0].1, test_data.embedding_with_single_tag);

        Ok(())
    }

    /// Tests that multiple tags in a group use AND logic (all required).
    #[sqlx::test(migrations = false)]
    async fn it_finds_embeddings_by_multiple_tags_and_logic(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        let test_data = seed_test_data(&pool).await?;

        let results = get_embeddings_by_tag_groups(&pool, &[&["tag_a", "tag_b"]]).await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[0].1, test_data.embedding_with_tags_a_and_b);

        Ok(())
    }

    /// Tests that multiple tag groups are handled separately, each with their own bin index.
    #[sqlx::test(migrations = false)]
    async fn it_handles_multiple_groups(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let test_data = seed_test_data(&pool).await?;

        let results = get_embeddings_by_tag_groups(&pool, &[&["group1"], &["group2"]]).await?;

        let group0: Vec<_> = results.iter().filter(|(bin, _)| *bin == 0).collect();
        let group1: Vec<_> = results.iter().filter(|(bin, _)| *bin == 1).collect();

        assert_eq!(group0.len(), 2);
        assert_eq!(group1.len(), 1);
        assert!(
            group0
                .iter()
                .any(|(_, id)| *id == test_data.embedding_group1_first)
        );
        assert!(
            group0
                .iter()
                .any(|(_, id)| *id == test_data.embedding_group1_second)
        );
        assert_eq!(group1[0].1, test_data.embedding_group2);

        Ok(())
    }

    /// Tests that non-existent tags return no results without error.
    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_for_missing_tags(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let _test_data = seed_test_data(&pool).await?;

        let results = get_embeddings_by_tag_groups(&pool, &[&["nonexistent_tag"]]).await?;

        assert!(results.is_empty());

        Ok(())
    }

    #[allow(dead_code)]
    struct TestData {
        embedding_with_single_tag: Uuid,
        embedding_with_other_tag: Uuid,
        embedding_with_tags_a_and_b: Uuid,
        embedding_with_only_tag_a: Uuid,
        embedding_group1_first: Uuid,
        embedding_group1_second: Uuid,
        embedding_group2: Uuid,
        embedding_no_tags: Uuid,
        embedding_unused: Uuid,
    }

    async fn seed_test_data(pool: &sqlx::PgPool) -> anyhow::Result<TestData> {
        crate::migrations::run_default_migrations(pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let encoder_digest =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")?;

        let embedding_with_single_tag = Uuid::parse_str("00000000-0000-0000-0000-000000000001")?;
        let embedding_with_other_tag = Uuid::parse_str("00000000-0000-0000-0000-000000000002")?;
        let embedding_with_tags_a_and_b = Uuid::parse_str("00000000-0000-0000-0000-000000000003")?;
        let embedding_with_only_tag_a = Uuid::parse_str("00000000-0000-0000-0000-000000000004")?;
        let embedding_group1_first = Uuid::parse_str("00000000-0000-0000-0000-000000000005")?;
        let embedding_group1_second = Uuid::parse_str("00000000-0000-0000-0000-000000000006")?;
        let embedding_group2 = Uuid::parse_str("00000000-0000-0000-0000-000000000007")?;
        let embedding_no_tags = Uuid::parse_str("00000000-0000-0000-0000-000000000008")?;
        let embedding_unused = Uuid::parse_str("00000000-0000-0000-0000-000000000009")?;

        let embeddings = vec![
            EmbeddingNew {
                id: embedding_with_single_tag,
                encoder_digest,
                encoded_at: now,
                value: [0.1; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_with_other_tag,
                encoder_digest,
                encoded_at: now,
                value: [0.2; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_with_tags_a_and_b,
                encoder_digest,
                encoded_at: now,
                value: [0.3; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_with_only_tag_a,
                encoder_digest,
                encoded_at: now,
                value: [0.4; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_group1_first,
                encoder_digest,
                encoded_at: now,
                value: [0.5; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_group1_second,
                encoder_digest,
                encoded_at: now,
                value: [0.6; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_group2,
                encoder_digest,
                encoded_at: now,
                value: [0.7; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_no_tags,
                encoder_digest,
                encoded_at: now,
                value: [0.8; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_unused,
                encoder_digest,
                encoded_at: now,
                value: [0.9; super::EMBEDDING_SIZE],
            },
        ];

        let tags = vec![
            TagNew::new("single_tag", embedding_with_single_tag, now),
            TagNew::new("other_tag", embedding_with_other_tag, now),
            TagNew::new("tag_a", embedding_with_tags_a_and_b, now),
            TagNew::new("tag_b", embedding_with_tags_a_and_b, now),
            TagNew::new("tag_a", embedding_with_only_tag_a, now),
            TagNew::new("group1", embedding_group1_first, now),
            TagNew::new("group1", embedding_group1_second, now),
            TagNew::new("group2", embedding_group2, now),
        ];

        store_embeddings(pool, &embeddings).await?;
        store_tags(pool, &tags).await?;

        Ok(TestData {
            embedding_with_single_tag,
            embedding_with_other_tag,
            embedding_with_tags_a_and_b,
            embedding_with_only_tag_a,
            embedding_group1_first,
            embedding_group1_second,
            embedding_group2,
            embedding_no_tags,
            embedding_unused,
        })
    }
}

#[derive(Debug, serde::Serialize, Clone)]
pub struct AggregatedDiversity {
    pub bin_index: i64,
    pub embedding_count: i64,
    pub min_knn_distance: Option<f64>,
    pub max_knn_distance: Option<f64>,
    pub avg_knn_distance: Option<f64>,
    pub sample_variance: Option<f64>,
    pub sample_std: Option<f64>,
}

/// Computes K-nearest neighbor distance statistics for pre-grouped embedding IDs.
///
/// Given an iterator of `(bin_index, embedding_id)` pairs, this function computes
/// K-NN distance statistics for all embeddings within each bin. It provides a measure
/// of local population density that can indicate convergence trends.
///
/// # Behavior
/// - **Empty input**: Returns an empty vector immediately if the input iterator is empty.
/// - **Invalid `k`**: If `k` is greater than or equal to the number of embeddings in a
///   bin, all distance metrics for that bin will be `None`. A `None` distance always
///   means "could not be computed" — either because the bin has fewer than 2 embeddings,
///   or because `k` exceeds the number of available neighbors. The caller can use the
///   returned `embedding_count` to distinguish between these two cases.
/// - **Bin isolation**: The K-NN search for an embedding is always scoped to its own
///   bin; embeddings in other bins are never considered as neighbors.
///
/// # Parameters
/// - `tx`: A database executor (generic over `PgExecutor` for transaction compatibility).
/// - `binned_ids`: An iterator of `(bin_index, embedding_id)` pairs that define the
///   population groups to analyze.
/// - `k`: The target neighbor to measure against. This is **1-indexed**, so `k=1`
///   selects the 1st nearest neighbor, `k=2` the 2nd, and so on.
///
/// # Returns
/// A vector of `AggregatedDiversity` structs, one for each distinct `bin_index` found
/// in the input. Results are ordered by `bin_index` ascending.
#[instrument(level = "debug", skip(tx, binned_ids), fields(k))]
pub async fn get_knn_diversity_stats<'tx, 'a, I, E>(
    tx: E,
    binned_ids: I,
    k: i32,
) -> Result<Vec<AggregatedDiversity>, super::Error>
where
    I: IntoIterator<Item = &'a (i64, Uuid)>,
    E: PgExecutor<'tx>,
{
    let (bin_indices, embedding_ids): (Vec<i64>, Vec<Uuid>) =
        binned_ids.into_iter().copied().unzip();

    if bin_indices.is_empty() {
        return Ok(Vec::new());
    }

    let rows = sqlx::query_as!(
        AggregatedDiversity,
        r#"
        WITH binned_embeddings_input AS (
            SELECT bin_index, embedding_id
            FROM UNNEST($1::BIGINT[], $2::UUID[]) AS t(bin_index, embedding_id)
        ),
        binned_embeddings AS (
            SELECT bei.bin_index, e.id AS embedding_id, e.value
            FROM binned_embeddings_input bei
            JOIN fx_durable_ga.embeddings e ON e.id = bei.embedding_id
        ),
        bin_counts AS (
            SELECT bin_index, COUNT(*) AS total
            FROM binned_embeddings
            GROUP BY bin_index
        ),
        kth_distances AS (
            SELECT
                t1.bin_index,
                t1.embedding_id,
                (
                    SELECT t2.value <=> t1.value
                    FROM binned_embeddings t2
                    WHERE t2.bin_index = t1.bin_index
                      AND t2.embedding_id <> t1.embedding_id
                    ORDER BY t2.value <=> t1.value
                    LIMIT 1 OFFSET $3::INT - 1
                ) AS knn_distance
            FROM binned_embeddings t1
            WHERE (SELECT bc.total FROM bin_counts bc WHERE bc.bin_index = t1.bin_index) > 1
        )
        SELECT
            bc.bin_index AS "bin_index!: i64",
            bc.total AS "embedding_count!: i64",
            MIN(kd.knn_distance) AS "min_knn_distance?: f64",
            MAX(kd.knn_distance) AS "max_knn_distance?: f64",
            AVG(kd.knn_distance) AS "avg_knn_distance?: f64",
            VAR_SAMP(kd.knn_distance) AS "sample_variance?: f64",
            STDDEV_SAMP(kd.knn_distance) AS "sample_std?: f64"
        FROM bin_counts bc
        LEFT JOIN kth_distances kd ON kd.bin_index = bc.bin_index
        GROUP BY bc.bin_index, bc.total
        ORDER BY bc.bin_index ASC
        "#,
        &bin_indices as &[i64],
        &embedding_ids as &[Uuid],
        k,
    )
    .fetch_all(tx)
    .await?;

    Ok(rows)
}

#[cfg(test)]
mod tests_get_knn_diversity_stats {
    use super::{EmbeddingNew, get_knn_diversity_stats, store_embeddings};
    use crate::services::indexing::Digest;
    use chrono::SubsecRound;
    use chrono::Utc;
    use uuid::Uuid;

    /// Empty input immediately returns no rows without hitting the database.
    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_for_no_input(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let results = get_knn_diversity_stats(&pool, &[], 1).await?;

        assert!(results.is_empty());

        Ok(())
    }

    /// A single embedding has no neighbors; all distance stats are None.
    #[sqlx::test(migrations = false)]
    async fn it_returns_none_distances_for_single_embedding(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        let data = seed_test_data(&pool).await?;

        let binned: Vec<(i64, Uuid)> = vec![(0, data.bin2_solo)];
        let results = get_knn_diversity_stats(&pool, &binned, 1).await?;
        let agg = &results[0];

        assert_eq!(results.len(), 1);
        assert_eq!(agg.bin_index, 0);
        assert_eq!(agg.embedding_count, 1);
        assert!(agg.min_knn_distance.is_none());
        assert!(agg.max_knn_distance.is_none());
        assert!(agg.avg_knn_distance.is_none());
        assert!(agg.sample_variance.is_none());
        assert!(agg.sample_std.is_none());

        Ok(())
    }

    /// Two embeddings in a bin: k=1 distance equals the distance between them.
    #[sqlx::test(migrations = false)]
    async fn it_computes_distance_for_two_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed_test_data(&pool).await?;

        let binned: Vec<(i64, Uuid)> = vec![(0, data.bin1_a), (0, data.bin1_b)];
        let results = get_knn_diversity_stats(&pool, &binned, 1).await?;
        let agg = &results[0];
        let avg = agg.avg_knn_distance.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(agg.embedding_count, 2);
        // Both embeddings are equidistant from each other, so min == max == avg.
        assert!(agg.avg_knn_distance.is_some());
        assert!((avg - agg.min_knn_distance.unwrap()).abs() < 1e-9);
        assert!((avg - agg.max_knn_distance.unwrap()).abs() < 1e-9);
        // Two embeddings each have the same k=1 distance, so VAR_SAMP across
        // those two identical values is 0.0 (not NULL).
        assert_eq!(agg.sample_variance.unwrap_or(0.0), 0.0);

        Ok(())
    }

    /// Three embeddings in bin 0: verifies count and that distance stats are populated.
    #[sqlx::test(migrations = false)]
    async fn it_computes_stats_for_bin_with_three_embeddings(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        let data = seed_test_data(&pool).await?;

        let binned: Vec<(i64, Uuid)> = vec![(0, data.bin0_a), (0, data.bin0_b), (0, data.bin0_c)];
        let results = get_knn_diversity_stats(&pool, &binned, 1).await?;
        let agg = &results[0];

        assert_eq!(results.len(), 1);
        assert_eq!(agg.bin_index, 0);
        assert_eq!(agg.embedding_count, 3);
        assert!(agg.min_knn_distance.is_some());
        assert!(agg.max_knn_distance.is_some());
        assert!(agg.avg_knn_distance.is_some());
        // With three distinct distances, sample variance should exist.
        assert!(agg.sample_variance.is_some());

        Ok(())
    }

    /// Multiple bins are computed independently and returned ordered by bin_index.
    #[sqlx::test(migrations = false)]
    async fn it_handles_multiple_bins_independently(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed_test_data(&pool).await?;

        let binned: Vec<(i64, Uuid)> = vec![
            (0, data.bin0_a),
            (0, data.bin0_b),
            (0, data.bin0_c),
            (1, data.bin1_a),
            (1, data.bin1_b),
            (2, data.bin2_solo),
        ];
        let results = get_knn_diversity_stats(&pool, &binned, 1).await?;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].bin_index, 0);
        assert_eq!(results[0].embedding_count, 3);
        assert_eq!(results[1].bin_index, 1);
        assert_eq!(results[1].embedding_count, 2);
        assert_eq!(results[2].bin_index, 2);
        assert_eq!(results[2].embedding_count, 1);
        assert!(results[2].avg_knn_distance.is_none()); // solo bin

        Ok(())
    }

    /// When k exceeds available neighbors, the knn_distance is None.
    #[sqlx::test(migrations = false)]
    async fn it_returns_none_when_k_is_out_of_bounds(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let data = seed_test_data(&pool).await?;

        // bin1 has only 2 embeddings but we ask for k=10
        let binned: Vec<(i64, Uuid)> = vec![(0, data.bin1_a), (0, data.bin1_b)];
        let results = get_knn_diversity_stats(&pool, &binned, 10).await?;
        let r = &results[0];

        assert_eq!(results.len(), 1);
        assert_eq!(r.embedding_count, 2);
        // Adaptive k should fall back to the only available neighbor.
        assert!(r.avg_knn_distance.is_none());

        Ok(())
    }

    struct TestData {
        /// bin 0: three embeddings placed on orthogonal unit-vector axes
        bin0_a: Uuid,
        bin0_b: Uuid,
        bin0_c: Uuid,
        /// bin 1: two embeddings placed at the same axis positions as bin0_a/bin0_b
        bin1_a: Uuid,
        bin1_b: Uuid,
        /// bin 2: exactly one embedding (no neighbors)
        bin2_solo: Uuid,
    }

    async fn seed_test_data(pool: &sqlx::PgPool) -> anyhow::Result<TestData> {
        crate::migrations::run_default_migrations(pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let encoder =
            Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")?;

        let build = |x: f32, y: f32| {
            let mut v = [0.0_f32; super::EMBEDDING_SIZE];
            v[0] = x;
            v[1] = y;
            v
        };

        let bin0_a = Uuid::parse_str("00000000-0000-0000-0000-000000000d01")?;
        let bin0_b = Uuid::parse_str("00000000-0000-0000-0000-000000000d02")?;
        let bin0_c = Uuid::parse_str("00000000-0000-0000-0000-000000000d03")?;
        let bin1_a = Uuid::parse_str("00000000-0000-0000-0000-000000000d04")?;
        let bin1_b = Uuid::parse_str("00000000-0000-0000-0000-000000000d05")?;
        let bin2_solo = Uuid::parse_str("00000000-0000-0000-0000-000000000d06")?;

        let embeddings = vec![
            EmbeddingNew {
                id: bin0_a,
                encoder_digest: encoder,
                encoded_at: now,
                value: build(1.0, 0.0),
            },
            EmbeddingNew {
                id: bin0_b,
                encoder_digest: encoder,
                encoded_at: now,
                value: build(0.0, 1.0),
            },
            EmbeddingNew {
                id: bin0_c,
                encoder_digest: encoder,
                encoded_at: now,
                value: build(1.0, 1.0),
            },
            EmbeddingNew {
                id: bin1_a,
                encoder_digest: encoder,
                encoded_at: now,
                value: build(1.0, 0.0),
            },
            EmbeddingNew {
                id: bin1_b,
                encoder_digest: encoder,
                encoded_at: now,
                value: build(0.0, 1.0),
            },
            EmbeddingNew {
                id: bin2_solo,
                encoder_digest: encoder,
                encoded_at: now,
                value: build(0.5, 0.5),
            },
        ];

        store_embeddings(pool, &embeddings).await?;

        Ok(TestData {
            bin0_a,
            bin0_b,
            bin0_c,
            bin1_a,
            bin1_b,
            bin2_solo,
        })
    }
}

#[derive(Debug, Default)]
pub struct SearchAssociatedTagsFilter {
    substring: Option<String>,
    tags: Option<Vec<String>>,
    cursor: Option<String>,
}

impl SearchAssociatedTagsFilter {
    pub fn with_substring(mut self, substring: String) -> Self {
        self.substring = Some(substring);
        self
    }

    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.get_or_insert_with(Vec::new).push(tag);
        self
    }

    pub fn with_cursor(mut self, cursor: String) -> Self {
        self.cursor = Some(cursor);
        self
    }
}

/// Searches for distinct tags that co-occur on the same embeddings as the given filter tags.
///
/// For each embedding that has at least one tag matching `filter.tags`, this query
/// collects all other tags on that embedding. Input tags are never included in the
/// returned results.
///
/// If `filter.substring` is provided, only associated tags whose name contains that
/// substring (case-insensitive) are returned. This is the primary way to efficiently
/// narrow results — for example, passing `"indexer_id:"` returns only tags with that
/// substring without fetching the full set of associated tags.
///
/// Results are paginated by tag name. Pass the cursor returned from the previous page
/// into `filter.cursor` to advance to the next page. A returned cursor of `None`
/// indicates the end of results — this occurs when the page returned fewer rows than
/// the requested limit.
///
/// # Returns
/// A tuple of `(tags, next_cursor)` where `next_cursor` is `Some(last_tag_name)` when
/// the page was full (more results may exist) and `None` when the final page has been
/// reached.
#[instrument(level = "debug", skip(tx))]
pub async fn search_distinct_associated_tags<'tx, E>(
    tx: E,
    filter: &SearchAssociatedTagsFilter,
    limit: i64,
) -> Result<(Vec<String>, Option<String>), super::Error>
where
    E: PgExecutor<'tx>,
{
    let rows = sqlx::query!(
        r#"
        SELECT DISTINCT et_assoc.tag_name
        FROM fx_durable_ga.embedding_tags et_filter
        JOIN fx_durable_ga.embedding_tags et_assoc ON et_filter.embedding_id = et_assoc.embedding_id
        WHERE
            ($1::TEXT[] IS NULL OR et_filter.tag_name = ANY($1::TEXT[]))
            AND ($2::TEXT IS NULL OR et_assoc.tag_name ILIKE '%' || $2 || '%')
            AND et_assoc.tag_name != ALL($1)
            AND ($3::TEXT IS NULL OR et_assoc.tag_name > $3)
        ORDER BY et_assoc.tag_name
        LIMIT $4
        "#,
        filter.tags.as_deref(),
        filter.substring.as_deref(),
        filter.cursor.as_deref(),
        limit
    )
    .fetch_all(tx)
    .await?;

    let next_cursor = if rows.len() == limit as usize {
        rows.last().map(|row| row.tag_name.clone())
    } else {
        None
    };
    let tags = rows.into_iter().map(|row| row.tag_name).collect();

    Ok((tags, next_cursor))
}

#[cfg(test)]
mod tests_search_distinct_associated_tags {
    use super::super::{EmbeddingNew, TagNew, queries::store_embeddings, queries::store_tags};
    use super::{SearchAssociatedTagsFilter, search_distinct_associated_tags};
    use crate::services::indexing::Digest;
    use chrono::{SubsecRound, Utc};
    use sqlx::PgPool;
    use uuid::Uuid;

    const ENCODER_HEX: &str = "00000000000000000000000000000000000000000000000000000000000000aa";

    struct TestData {
        _embedding_red: Uuid,
        _embedding_red_2: Uuid,
        _embedding_blue: Uuid,
    }

    async fn seed(pool: &PgPool) -> anyhow::Result<TestData> {
        crate::migrations::run_default_migrations(pool).await?;

        let now = Utc::now().trunc_subsecs(6);
        let encoder = Digest::from_hex(ENCODER_HEX)?;

        let embedding_red = Uuid::parse_str("00000000-0000-0000-0000-000000000a01")?;
        let embedding_red_2 = Uuid::parse_str("00000000-0000-0000-0000-000000000a02")?;
        let embedding_blue = Uuid::parse_str("00000000-0000-0000-0000-000000000a03")?;

        let embeddings = vec![
            EmbeddingNew {
                id: embedding_red,
                encoder_digest: encoder,
                encoded_at: now,
                value: [0.1; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_red_2,
                encoder_digest: encoder,
                encoded_at: now,
                value: [0.2; super::EMBEDDING_SIZE],
            },
            EmbeddingNew {
                id: embedding_blue,
                encoder_digest: encoder,
                encoded_at: now,
                value: [0.3; super::EMBEDDING_SIZE],
            },
        ];

        store_embeddings(pool, &embeddings).await?;

        let tags = vec![
            TagNew::new("red", embedding_red, now),
            TagNew::new("round", embedding_red, now),
            TagNew::new("fruit", embedding_red, now),
            TagNew::new("red", embedding_red_2, now),
            TagNew::new("sweet", embedding_red_2, now),
            TagNew::new("berry", embedding_red_2, now),
            TagNew::new("blue", embedding_blue, now),
            TagNew::new("round", embedding_blue, now),
            TagNew::new("berry", embedding_blue, now),
        ];

        store_tags(pool, &tags).await?;

        Ok(TestData {
            _embedding_red: embedding_red,
            _embedding_red_2: embedding_red_2,
            _embedding_blue: embedding_blue,
        })
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_distinct_associated_tags(pool: PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let filter = SearchAssociatedTagsFilter::default().with_tag("red".to_string());
        let (tags, _) = search_distinct_associated_tags(&pool, &filter, 10).await?;

        assert_eq!(tags, vec!["berry", "fruit", "round", "sweet"]);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_excludes_input_tags(pool: PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let filter = SearchAssociatedTagsFilter::default()
            .with_tag("red".to_string())
            .with_tag("round".to_string());
        let (tags, _) = search_distinct_associated_tags(&pool, &filter, 10).await?;

        assert!(!tags.contains(&"red".to_string()));
        assert!(!tags.contains(&"round".to_string()));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_substring(pool: PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let filter = SearchAssociatedTagsFilter::default()
            .with_tag("red".to_string())
            .with_substring("be".to_string());
        let (tags, _) = search_distinct_associated_tags(&pool, &filter, 10).await?;

        assert_eq!(tags, vec!["berry"]);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_for_no_match(pool: PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let filter = SearchAssociatedTagsFilter::default().with_tag("nonexistent".to_string());
        let (tags, _) = search_distinct_associated_tags(&pool, &filter, 10).await?;

        assert!(tags.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_paginates_with_cursor(pool: PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let filter = SearchAssociatedTagsFilter::default().with_tag("red".to_string());
        let (first_page, cursor) = search_distinct_associated_tags(&pool, &filter, 2).await?;

        assert_eq!(first_page.len(), 2);
        assert!(cursor.is_some());

        let cursor_filter = SearchAssociatedTagsFilter::default()
            .with_tag("red".to_string())
            .with_cursor(cursor.unwrap());
        let (second_page, _) = search_distinct_associated_tags(&pool, &cursor_filter, 10).await?;

        assert_eq!(second_page.len(), 2);
        assert_eq!(first_page, vec!["berry", "fruit"]);
        assert_eq!(second_page, vec!["round", "sweet"]);

        Ok(())
    }
}
