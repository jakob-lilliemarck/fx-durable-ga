use std::fmt::Debug;

use super::{
    Embedding, EmbeddingNew, RequestedEmbedding, SearchEmbeddingsFilter,
    SearchRequestedEmbeddingsFilter, SearchSimilarEmbeddingsFilter, TagNew,
    queries::{AggregatedDiversity, SearchAssociatedTagsFilter},
};
use crate::{
    infrastructure::db::{self, ReadPool, WritePool},
    services::indexing::Digest,
};
use chrono::{DateTime, Utc};
use sqlx::PgTransaction;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Read {
    ro: ReadPool,
}

#[derive(Debug, Clone)]
pub struct Write {
    wr: WritePool,
}

pub struct WriteTx<'tx> {
    tx: &'tx mut PgTransaction<'static>,
}

impl db::Tx for Write {
    type Error = super::Error;

    fn tx(self) -> db::TxFut<Self::Error> {
        let pool = self.wr.pool.clone();
        Box::pin(async move {
            let tx = pool.begin().await?;
            Ok(tx)
        })
    }
}

impl Read {
    pub fn new(ro: ReadPool) -> Self {
        Self { ro }
    }

    #[instrument(level = "debug", skip(self))]
    pub async fn find_similar(
        &self,
        encoder_digest: &Digest,
        filter: &SearchSimilarEmbeddingsFilter,
        limit: i64,
    ) -> Result<Vec<(Embedding, f64)>, super::Error> {
        super::queries::search_similar_embeddings(&self.ro.pool, encoder_digest, filter, limit)
            .await
    }

    #[instrument(level = "debug", skip(self))]
    pub async fn search_embeddings(
        &self,
        filter: &SearchEmbeddingsFilter,
        limit: i64,
    ) -> Result<Vec<Embedding>, super::Error> {
        super::queries::search_embeddings(&self.ro.pool, filter, limit).await
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn search_requested_embeddings(
        &self,
        filter: &SearchRequestedEmbeddingsFilter,
        limit: i64,
    ) -> Result<Vec<RequestedEmbedding>, super::Error> {
        super::queries::search_requested_embeddings(&self.ro.pool, filter, limit).await
    }

    // FIXME!
    // Rename this function and write some proper docs
    pub async fn get_tag_pairs_for_embeddings(
        &self,
        tags_lhs: &[String],
        tags_rhs: &[String],
    ) -> Result<Vec<(Uuid, String, String)>, super::Error> {
        super::queries::get_tag_pairs_for_embeddings(&self.ro.pool, tags_lhs, tags_rhs).await
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn get_embeddings_by_tag_groups(
        &self,
        tag_groups: &[&[&str]],
    ) -> Result<Vec<(i64, Uuid)>, super::Error> {
        super::queries::get_embeddings_by_tag_groups(&self.ro.pool, tag_groups).await
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn get_knn_diversity_stats<'a, I>(
        &self,
        binned_ids: I,
        k: i32,
    ) -> Result<Vec<AggregatedDiversity>, super::Error>
    where
        I: IntoIterator<Item = &'a (i64, Uuid)> + Debug,
    {
        super::queries::get_knn_diversity_stats(&self.ro.pool, binned_ids, k).await
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn get_distinct_associated_tags(
        &self,
        filter: &SearchAssociatedTagsFilter,
        limit: i64,
    ) -> Result<(Vec<String>, Option<String>), super::Error> {
        super::queries::search_distinct_associated_tags(&self.ro.pool, filter, limit).await
    }
}

impl Write {
    pub fn new(wr: WritePool) -> Self {
        Self { wr }
    }
}

impl<'tx> WriteTx<'tx> {
    #[instrument(level = "debug", skip(tx))]
    pub fn new(tx: &'tx mut PgTransaction<'static>) -> Self {
        Self { tx }
    }

    #[instrument(level = "debug", skip(self, embeddings))]
    pub async fn store_embeddings<'a, I>(
        &mut self,
        embeddings: I,
    ) -> Result<Vec<Uuid>, super::Error>
    where
        I: IntoIterator<Item = &'a EmbeddingNew>,
    {
        super::queries::store_embeddings(&mut **self.tx, embeddings).await
    }

    #[instrument(level = "debug", skip(self, tags))]
    pub async fn store_tags<'a, I>(&mut self, tags: I) -> Result<Vec<(Uuid, String)>, super::Error>
    where
        I: IntoIterator<Item = &'a TagNew>,
    {
        super::queries::store_tags(&mut **self.tx, tags).await
    }

    #[instrument(level = "debug", skip(self, requested_embeddings))]
    pub(crate) async fn store_requested_embeddings<'a, I>(
        &mut self,
        requested_embeddings: I,
    ) -> Result<Vec<RequestedEmbedding>, super::Error>
    where
        I: IntoIterator<Item = &'a RequestedEmbedding>,
    {
        super::queries::store_requested_embeddings(&mut **self.tx, requested_embeddings).await
    }

    #[instrument(level = "debug", skip(self, requested_embeddings))]
    pub(crate) async fn set_requested_embeddings_handled_at<'a, I>(
        &mut self,
        requested_embeddings: I,
        handled_at: DateTime<Utc>,
    ) -> Result<u64, super::Error>
    where
        I: IntoIterator<Item = &'a (Uuid, Digest)>,
    {
        super::queries::set_requested_embeddings_handled_at(
            &mut **self.tx,
            requested_embeddings,
            handled_at,
        )
        .await
    }
}
