use uuid::Uuid;

use crate::gene::{Gene, Genotype};

/// Genotype repository
pub struct Repository {}

#[derive(Debug, thiserror::Error)]
pub enum Error {}

impl Repository {
    pub(crate) async fn new_genotype(
        &self,
        genes: &[Gene],
        morphology_id: Uuid,
    ) -> Result<Genotype, Error> {
        todo!()
    }

    pub(crate) async fn get_genotype(&self, id: Uuid) -> Result<Genotype, Error> {
        todo!()
    }

    pub(crate) async fn set_fitness(&self, id: Uuid) -> Result<(), Error> {
        todo!()
    }
}
