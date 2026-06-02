use crate::configuration::EnableListening;
use crate::infrastructure::db;
use crate::infrastructure::di::{Container, InvokeError, InvokeResult};
use crate::services::indexing::{self as indexable, Indexer};
use crate::services::optimization::Optimizer;
use crate::services::optimization::OptimizerRegistry;
use futures::future::BoxFuture;
use futures::lock::Mutex;
use sqlx::PgPool;
use std::sync::{Arc, Once};
use tracing::Level;
use tracing_subscriber;

/// Builder for test DI containers. Listeners are disabled by default.
pub struct TestConfig {
    pool: PgPool,
    listeners: bool,
    invokables: Vec<
        Box<dyn for<'a> FnOnce(&'a mut Container) -> BoxFuture<'a, InvokeResult> + Send + Sync>,
    >,
}

impl TestConfig {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            listeners: false,
            invokables: Vec::new(),
        }
    }

    pub fn with_listeners(mut self) -> Self {
        self.listeners = true;
        self
    }

    pub fn with_indexer<I>(mut self, indexer: I) -> Self
    where
        I: Indexer + 'static,
    {
        self.invokables.push(Box::new(move |c: &mut Container| {
            Box::pin(async move {
                let provided = c.get::<Arc<Mutex<indexable::Registry>>>().await?;
                let mut lock = provided.lock().await;
                lock.register(Arc::new(indexer))
                    .map_err(|err| InvokeError::new(err))?;
                Ok(())
            })
        }));
        self
    }

    pub fn with_optimizer<O>(mut self, optimizer: O) -> Self
    where
        O: Optimizer + 'static,
    {
        self.invokables.push(Box::new(move |c: &mut Container| {
            Box::pin(async move {
                let provided = c.get::<Arc<Mutex<OptimizerRegistry>>>().await?;
                let mut lock = provided.lock().await;
                lock.register(optimizer);
                Ok(())
            })
        }));
        self
    }

    pub async fn build(self) -> anyhow::Result<Container> {
        let mut c = Container::new();

        crate::register(&mut c);

        let listeners = self.listeners;
        c.provide(move |_| Box::pin(async move { Ok(EnableListening { value: listeners }) }));
        let pool = self.pool;
        let ro = pool.clone();
        c.provide(move |_| Box::pin(async move { Ok(db::ReadPool { pool: ro }) }));
        let wr = pool.clone();
        c.provide(move |_| Box::pin(async move { Ok(db::WritePool { pool: wr }) }));

        for invokable in self.invokables {
            c.invokable(invokable);
        }

        c.invoke().await?;

        Ok(c)
    }
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
