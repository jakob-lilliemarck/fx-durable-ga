use crate::gene::Population;

/// Populations repository
pub struct Repository {}

#[derive(Debug, thiserror::Error)]
pub enum Error {}

impl Repository {
    pub(crate) async fn new_population(&self) -> Result<Population, Error> {
        todo!()
    }

    pub(crate) async fn get_population(&self) -> Result<Population, Error> {
        todo!()
    }
}
