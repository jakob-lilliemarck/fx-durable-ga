use crate::models::mutagen::Mutagen;

use super::{FitnessGoal, Strategy};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub(crate) struct Request {
    pub(crate) id: Uuid,
    pub(crate) requested_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) goal: FitnessGoal,
    pub(crate) threshold: f64,
    pub(crate) strategy: Strategy,
    pub(crate) mutagen: Mutagen,
}
