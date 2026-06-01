use crate::bootstrap::App;
use crate::configuration::EnableListening;
use crate::infrastructure::db;
use crate::infrastructure::di::{Container, InvokeError};
use crate::services::indexing::{self as indexable, Indexer};
use crate::services::optimization::Optimizer;
use crate::services::optimization::OptimizerRegistry;
use futures::lock::Mutex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use sqlx::PgPool;
use std::sync::{Arc, Once};
use tracing::Level;
use tracing_subscriber;

pub async fn create_test_app_builder<T, O, I>(
    pool: PgPool,
    type_name: &'static str,
    opt: O,
    ind: I,
) -> anyhow::Result<Arc<App>>
where
    T: Serialize + DeserializeOwned + Send + Sync + 'static,
    O: Optimizer<Type = T> + 'static,
    I: Indexer<Type = T> + 'static,
{
    // Create a DI container
    let mut c = Container::new();

    // Invoke optimizer registration
    c.invokable(move |c| {
        Box::pin(async move {
            let provided = c.get::<Arc<Mutex<OptimizerRegistry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(type_name, opt);
            Ok(())
        })
    });

    // Invoke indexer registration
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<Mutex<indexable::Registry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(Arc::new(ind))
                .map_err(|err| InvokeError::new(err))?;
            Ok(())
        })
    });

    // Register fx-durable-ga with the DI container
    crate::register(&mut c);

    // Overwrite with the provided pool
    let wr = pool.clone();
    c.provide(|_| Box::pin(async { Ok(db::WritePool { pool: wr }) }));
    let ro = pool.clone();
    c.provide(|_| Box::pin(async { Ok(db::ReadPool { pool: ro }) }));

    // Invoke all
    c.invoke().await?;

    let app = c.get::<Arc<App>>().await?;

    Ok(app)
}

/// Creates a DI container wired with all service registrations and the given test pool.
/// Listening is disabled so invokables short-circuit. Callers extract the service they need.
pub async fn create_test_container(pool: PgPool) -> anyhow::Result<Container> {
    let mut c = Container::new();

    crate::register(&mut c);

    c.provide(|_| Box::pin(async { Ok(EnableListening { value: false }) }));
    let ro = pool.clone();
    c.provide(|_| Box::pin(async { Ok(db::ReadPool { pool: ro }) }));
    let wr = pool.clone();
    c.provide(|_| Box::pin(async { Ok(db::WritePool { pool: wr }) }));

    c.invoke().await?;

    Ok(c)
}

static TEST_TRACING_SUBSCRIBER: Once = Once::new();

pub fn init_test_tracing() {
    TEST_TRACING_SUBSCRIBER.call_once(|| {
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_max_level(Level::INFO)
            .init();
    });
}
