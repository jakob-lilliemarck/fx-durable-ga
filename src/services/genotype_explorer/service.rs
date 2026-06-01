use super::errors::Error;
use crate::repositories::genotypes;
use crate::repositories::genotypes::Genotype;
use tracing::instrument;
use uuid::Uuid;

pub struct Service {
    pub(super) genotypes_ro: genotypes::Read,
}

impl Service {
    pub fn new(genotypes_ro: genotypes::Read) -> Self {
        Self { genotypes_ro }
    }

    #[instrument(level = "debug", skip(self), fields(genotype_id = %genotype_id, degree=degree))]
    pub async fn get_ancestors(
        &self,
        genotype_id: &Uuid,
        degree: u32,
    ) -> Result<Vec<Genotype>, Error> {
        if degree > i32::MAX as u32 {
            return Err(Error::DegreeOverflow { degree });
        };

        let ancestors = self
            .genotypes_ro
            .get_ancestors(genotype_id, degree as i32)
            .await?;

        Ok(ancestors)
    }

    #[instrument(level = "debug", skip(self), fields(genotype_id = %genotype_id, degree=degree))]
    pub async fn get_descendants(
        &self,
        genotype_id: &Uuid,
        degree: u32,
    ) -> Result<Vec<Genotype>, Error> {
        if degree > i32::MAX as u32 {
            return Err(Error::DegreeOverflow { degree });
        };

        let descendants = self
            .genotypes_ro
            .get_descendants(genotype_id, degree as i32)
            .await?;

        Ok(descendants)
    }

    /// DEPRECATED: Use genotypes repository directly instead
    #[deprecated]
    #[instrument(level = "debug", skip(self), fields(type_name))]
    pub async fn search_genotypes(
        &self,
        filter: &genotypes::SearchGenotypesFilter,
        limit: i64,
    ) -> Result<Vec<Genotype>, Error> {
        let genotypes = self.genotypes_ro.search_genotypes(filter, limit).await?;
        Ok(genotypes)
    }

}
