#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Missing environment variable \"{key}\"\n\tMessage: {message}")]
    Missing { key: String, message: String },
    #[error("Could not parse environment variable: {key}\n\tGot: {value}\n\tMessage: {message}")]
    Invalid {
        key: String,
        value: String,
        message: String,
    },
}

pub trait Var {
    const NAME: &'static str;
    type Type;

    fn from_env() -> Result<Self::Type, ConfigError>;
}

pub struct ApiBaseUrl;

impl Var for ApiBaseUrl {
    const NAME: &'static str = "API_BASE_URL";
    type Type = String;

    fn from_env() -> Result<Self::Type, ConfigError> {
        let val = std::env::var(Self::NAME).map_err(|err| ConfigError::Missing {
            key: Self::NAME.to_string(),
            message: err.to_string(),
        })?;

        Ok(val)
    }
}
