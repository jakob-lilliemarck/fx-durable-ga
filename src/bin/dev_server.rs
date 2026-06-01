use fx_durable_ga::{bootstrap::App, configuration::BindAddr};
use std::sync::Arc;
use tracing::{Level, info};

#[derive(Debug, thiserror::Error)]
#[error("No bind address configured")]
pub struct NoBindAddr;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::from_filename(".env.local").ok();
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    // Create a DI container
    let mut c = fx_durable_ga::infrastructure::di::Container::new();

    // Register fx-durable-ga with the container
    fx_durable_ga::register(&mut c);

    // Perform any desired overwrites

    // Get the app instance from the container
    let app = c.get::<Arc<App>>().await?;

    let Some(bind_addr) = c.get::<Option<BindAddr>>().await? else {
        return Err(NoBindAddr.into());
    };

    info!("http://{}/docs", bind_addr.value);
    info!("http://{}/lineage", bind_addr.value);
    info!("http://{}/optimizations", bind_addr.value);

    App::serve_http(app, bind_addr.value).await?;

    Ok(())
}
