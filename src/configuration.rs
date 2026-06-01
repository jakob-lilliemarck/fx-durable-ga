use crate::infrastructure::di::{Container, ProviderError, ProviderResult};
use futures::future::BoxFuture;
use std::{net::SocketAddr, str::FromStr, time::Duration};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Missing environment variable \"{key}\"")]
    Missing { key: &'static str },

    #[error("Invalid environment variable value at \"{key}\"")]
    Invalid { key: &'static str },

    #[error("Failed to parse environment variable \"{key}\" to type {type_name}")]
    Parse {
        key: &'static str,
        type_name: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    #[error("Could not parse environment variable: {key}\n\tGot: {value}\n\tMessage: {message}")]
    Validation {
        key: String,
        value: String,
        message: String,
    },
}

type ConfigResult<T> = Result<T, Error>;

trait Var {
    const NAME: &'static str;
    type Type;

    fn from_env() -> Result<Self::Type, Error>;
}

fn get_optional<T: FromStr>(key: &'static str) -> ConfigResult<Option<T>>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    match std::env::var(key) {
        Ok(value) => value.parse::<T>().map(Some).map_err(|err| Error::Parse {
            key,
            type_name: std::any::type_name::<T>(),
            source: Box::new(err),
        }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::Invalid { key }),
    }
}

fn get_required<T: FromStr>(key: &'static str) -> ConfigResult<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    get_optional(key)?.ok_or(Error::Missing { key })
}

#[derive(Clone)]
pub struct HostId {
    pub value: Uuid,
}

#[derive(Clone)]
pub struct DatabaseReadUrl {
    pub value: String,
}

#[derive(Clone)]
pub struct DatabaseWriteUrl {
    pub value: String,
}

#[derive(Clone)]
pub struct MaxDeduplicationAttempts {
    pub value: i32,
}

#[derive(Clone)]
pub struct JobWorkerCount {
    pub value: usize,
}

#[derive(Clone)]
pub struct PollIntervalSeconds {
    pub value: u64,
}

#[derive(Clone)]
pub struct JobLeaseDuration {
    pub value: Duration,
}

// Used by the server to listen for incoming requests
#[derive(Clone)]
pub struct BindAddr {
    pub value: SocketAddr,
}

// Used by the client to configure the url of the remote
#[derive(Clone)]
pub struct ApiBaseUrl {
    value: String,
}

#[derive(Clone)]
pub struct EnableListening {
    pub value: bool,
}

impl Var for HostId {
    const NAME: &'static str = "HOST_ID";
    type Type = HostId;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_required(Self::NAME)?;
        Ok(HostId { value })
    }
}

impl Var for DatabaseReadUrl {
    const NAME: &'static str = "DATABASE_URL";
    type Type = DatabaseReadUrl;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_required(Self::NAME)?;
        Ok(DatabaseReadUrl { value })
    }
}

impl Var for DatabaseWriteUrl {
    const NAME: &'static str = "DATABASE_WRITE_URL";
    type Type = DatabaseWriteUrl;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_required(Self::NAME)?;
        Ok(DatabaseWriteUrl { value })
    }
}

impl Var for MaxDeduplicationAttempts {
    const NAME: &'static str = "MAX_DEDUPLICATION_ATTEMPTS";
    type Type = MaxDeduplicationAttempts;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_required(Self::NAME)?;
        Ok(MaxDeduplicationAttempts { value })
    }
}

impl Var for JobWorkerCount {
    const NAME: &'static str = "JOB_WORKER_COUNT";
    type Type = JobWorkerCount;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_required(Self::NAME)?;
        Ok(JobWorkerCount { value })
    }
}

impl Var for JobLeaseDuration {
    const NAME: &'static str = "JOB_LEASE_SECONDS";
    type Type = JobLeaseDuration;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_required(Self::NAME)?;

        Ok(JobLeaseDuration {
            value: Duration::from_secs(value),
        })
    }
}

// shared value used across semaphores, jobs and events
impl Var for PollIntervalSeconds {
    const NAME: &'static str = "LISTENER_POLL_INTERVAL_SECONDS";
    type Type = PollIntervalSeconds;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_optional(Self::NAME)?.unwrap_or(60);
        Ok(PollIntervalSeconds { value })
    }
}

impl Var for BindAddr {
    const NAME: &'static str = "BIND_ADDR";
    type Type = Option<BindAddr>;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_optional(Self::NAME)?.map(|value| BindAddr { value });
        Ok(value)
    }
}

impl Var for ApiBaseUrl {
    const NAME: &'static str = "API_BASE_URL";
    type Type = ApiBaseUrl;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_required(Self::NAME)?;
        Ok(ApiBaseUrl { value })
    }
}

impl Var for EnableListening {
    const NAME: &'static str = "ENABLE_LISTENING";
    type Type = EnableListening;

    fn from_env() -> Result<Self::Type, Error> {
        let value = get_optional(Self::NAME)?.unwrap_or(true);
        Ok(EnableListening { value })
    }
}

fn provide_conf<T: Var>(_: &mut Container) -> BoxFuture<'_, ProviderResult<T::Type>> {
    Box::pin(async { T::from_env().map_err(|err| ProviderError::new::<T::Type, _>(Box::new(err))) })
}

pub fn register(c: &mut Container) {
    c.provide(provide_conf::<HostId>);
    c.provide(provide_conf::<DatabaseReadUrl>);
    c.provide(provide_conf::<DatabaseWriteUrl>);
    c.provide(provide_conf::<MaxDeduplicationAttempts>);
    c.provide(provide_conf::<PollIntervalSeconds>);
    c.provide(provide_conf::<JobWorkerCount>);
    c.provide(provide_conf::<JobLeaseDuration>);
    c.provide(provide_conf::<BindAddr>);
    c.provide(provide_conf::<ApiBaseUrl>);
    c.provide(provide_conf::<EnableListening>);
}
