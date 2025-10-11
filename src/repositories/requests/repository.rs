/// Optimization requests repository
pub struct Repository {}

#[derive(Debug, thiserror::Error)]
pub enum Error {}

impl Repository {
    pub fn new_request(&self) -> Result<(), Error> {
        todo!()
    }

    pub fn get_request(&self) -> Result<(), Error> {
        todo!()
    }
}
