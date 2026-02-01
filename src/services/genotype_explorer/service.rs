use super::errors::Error;
use crate::{
    models::{Evaluation, Genotype, TimingsSummary},
    repositories::genotypes,
};
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

pub struct Service {
    pub(super) genotypes: Arc<genotypes::Repository>,
}

impl Service {
    #[instrument(level = "debug", skip(genotypes))]
    pub(crate) fn new(genotypes: &Arc<genotypes::Repository>) -> Self {
        Self {
            genotypes: genotypes.clone(),
        }
    }

    #[instrument(level = "debug", skip(self), fields(genotype_id = %genotype_id, degree=degree))]
    pub async fn get_ancestors(
        &self,
        genotype_id: &Uuid,
        degree: u32,
    ) -> Result<Vec<(Genotype, Evaluation)>, Error> {
        if degree > i32::MAX as u32 {
            return Err(Error::DegreeOverflow { degree });
        };

        let ancestors = self
            .genotypes
            .get_ancestors(genotype_id, degree as i32)
            .await?;

        Ok(ancestors)
    }

    #[instrument(level = "debug", skip(self), fields(genotype_id = %genotype_id, degree=degree))]
    pub async fn get_descendants(
        &self,
        genotype_id: &Uuid,
        degree: u32,
    ) -> Result<Vec<(Genotype, Evaluation)>, Error> {
        if degree > i32::MAX as u32 {
            return Err(Error::DegreeOverflow { degree });
        };

        let ancestors = self
            .genotypes
            .get_descendants(genotype_id, degree as i32)
            .await?;

        Ok(ancestors)
    }

    #[instrument(level = "debug", skip(self), fields(type_name))]
    pub async fn search_genotypes(
        &self,
        filter: &genotypes::SearchFilter,
        limit: i64,
    ) -> Result<Vec<(Genotype, Option<f64>)>, Error> {
        let genotypes = self.genotypes.search_genotypes(filter, limit).await?;
        Ok(genotypes)
    }

    #[instrument(level = "debug", skip(self))]
    pub async fn get_timing_summary<'a>(
        &self,
        filter: &genotypes::GetTimingsFilter<'a>,
        percentiles: &[f64],
    ) -> Result<TimingsSummary, Error> {
        let timings = self.genotypes.get_timings(filter, percentiles).await?;
        Ok(timings)
    }
}
