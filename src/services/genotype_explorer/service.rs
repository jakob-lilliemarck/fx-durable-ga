use super::errors::Error;
use crate::repositories::genotypes;
use crate::repositories::genotypes::Genotype;
use tracing::instrument;
use uuid::Uuid;

/// Exploration service for navigating genotype ancestry and descendants.
pub struct Service {
    pub(super) genotypes_ro: genotypes::Read,
}

impl Service {
    /// Creates a new genotype explorer service.
    pub fn new(genotypes_ro: genotypes::Read) -> Self {
        Self { genotypes_ro }
    }

    /// Returns ancestor genotypes up to the given degree from the specified genotype.
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

    /// Returns descendant genotypes up to the given degree from the specified genotype.
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
}

#[cfg(test)]
mod tests {
    use super::Service;
    use crate::repositories::genotypes::Genotype;
    use crate::repositories::genotypes::store_genotypes;
    use crate::services::optimization::{self, store_request};
    use crate::test_tools::TestConfig;
    use std::sync::Arc;
    use uuid::Uuid;

    /// Seeds a simple lineage: root → child → grandchild
    async fn seed(pool: &sqlx::PgPool) -> anyhow::Result<(Uuid, Uuid, Uuid)> {
        let request = optimization::Request::new(
            "test",
            optimization::FitnessGoal::maximize(1.0).unwrap(),
            optimization::Selector::tournament(3),
            optimization::Schedule::generational(10, 2),
        );
        let request_id = request.id;
        store_request(pool, request).await?;

        let root = Genotype::new(
            "test",
            serde_json::json!([0]),
            request_id,
            Some(1),
            None,
            None,
        )?;
        let root_id = root.id();

        let child = Genotype::new(
            "test",
            serde_json::json!([1]),
            request_id,
            Some(2),
            Some(&root_id),
            None,
        )?;
        let child_id = child.id();

        let grandchild = Genotype::new(
            "test",
            serde_json::json!([2]),
            request_id,
            Some(3),
            Some(&child_id),
            None,
        )?;
        let grandchild_id = grandchild.id();

        store_genotypes(pool, &[root, child, grandchild]).await?;

        Ok((root_id, child_id, grandchild_id))
    }

    async fn build_service(pool: &sqlx::PgPool) -> anyhow::Result<Arc<Service>> {
        let mut c = TestConfig::new(pool.clone()).build().await?;
        let svc = c.get::<Arc<Service>>().await?;
        Ok(svc)
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_ancestors(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (root_id, child_id, grandchild_id) = seed(&pool).await?;

        let svc = build_service(&pool).await?;

        let ancestors = svc.get_ancestors(&grandchild_id, 2).await?;
        assert_eq!(ancestors.len(), 3);
        assert_eq!(ancestors[0].id(), grandchild_id);
        assert_eq!(ancestors[1].id(), child_id);
        assert_eq!(ancestors[2].id(), root_id);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_descendants(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (root_id, child_id, grandchild_id) = seed(&pool).await?;

        let svc = build_service(&pool).await?;

        let descendants = svc.get_descendants(&root_id, 2).await?;
        assert_eq!(descendants.len(), 3);
        assert_eq!(descendants[0].id(), root_id);
        assert_eq!(descendants[1].id(), child_id);
        assert_eq!(descendants[2].id(), grandchild_id);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_degree_overflow(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (root_id, _, _) = seed(&pool).await?;

        let svc = build_service(&pool).await?;

        let overflow = i32::MAX as u32 + 1;
        let result = svc.get_ancestors(&root_id, overflow).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), super::Error::DegreeOverflow { degree } if degree == overflow)
        );

        Ok(())
    }
}
