use super::repositories::semaphores::{self, Semaphore, models::SEMAPHORE_CHANNEL};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use futures::future::{Either, pending};
use fx_pgmux::Multiplexer;
use sqlx::PgTransaction;
use std::{sync::Arc, time::Duration};
use tokio::sync::RwLock;
use tokio::{
    sync::{broadcast, oneshot},
    task::JoinHandle,
    time::Instant,
};
use tracing::instrument;

const BUFFER: usize = 100;

/// Handle to control and synchronize with the notification agent.
pub struct Handle {
    tx: broadcast::Sender<semaphores::Semaphore>,
    tx_stop: oneshot::Sender<()>,
    handle: JoinHandle<Result<(), super::Error>>,
}

impl Handle {
    /// Stops the agent and waits for it to finish.
    #[instrument(level = "debug", skip(self))]
    async fn stop(self) -> Result<(), super::Error> {
        // Send stop signal to Agent
        self.tx_stop
            .send(())
            .map_err(|_| super::Error::OneshotSend)?;

        // Wait for Agent to finish
        self.handle.await??;

        Ok(())
    }

    /// Subscribes to semaphore notifications from the agent.
    fn subscribe(&self) -> broadcast::Receiver<semaphores::Semaphore> {
        self.tx.subscribe()
    }
}

/// Synchronization service provides one to many synchronization, intended for process
/// management across a distributed system.
pub struct Service {
    handle: RwLock<Option<Handle>>,
    semaphores: Arc<semaphores::Repository>,
    poll_interval: Option<Duration>,
}

impl Service {
    /// Creates a new synchronization service.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn new(
        poll_interval: Option<Duration>,
        semaphores: Arc<super::repositories::semaphores::Repository>,
    ) -> Self {
        tracing::debug!(
            message = "creating synchronization service",
            poll_interval = ?poll_interval
        );

        Self {
            semaphores,
            handle: RwLock::new(None),
            poll_interval,
        }
    }

    /// Registers the service with a multiplexer and starts the notification agent.
    #[instrument(level = "debug", skip(self, mux))]
    pub async fn register<'a>(&self, mux: &'a mut Multiplexer) -> Result<(), super::Error> {
        let stream = mux.register(SEMAPHORE_CHANNEL).await?;

        let (tx, _) = broadcast::channel(BUFFER);
        let (tx_stop, rx_stop) = oneshot::channel::<()>();
        let agent = Agent::new(stream, &self.semaphores, &tx, rx_stop, &self.poll_interval);
        let handle = run_agent(agent);

        let handle = Handle {
            handle,
            tx_stop,
            tx,
        };

        {
            let mut write = self.handle.write().await;
            *write = Some(handle);
        }

        Ok(())
    }

    /// Waits for a semaphore to be raised, returning immediately if already raised.
    #[instrument(level = "info", skip(self))]
    pub async fn wait_for(&self, name: &str, since: &DateTime<Utc>) -> Result<(), super::Error> {
        let mut rx = {
            let read = self.handle.read().await;
            let Some(ref handle) = *read else {
                return Err(super::Error::Uninitialized);
            };
            handle.subscribe()
        };

        // If any semaphore as been raised since the specified time, return early
        let filter = &semaphores::PollFilter::default().with_name(name.to_string());
        let semaphores = self.semaphores.poll(since, &filter).await?;
        if !semaphores.is_empty() {
            return Ok(());
        }

        loop {
            match rx.recv().await {
                Ok(semaphore) => {
                    // If the semaphore name does not match, ignore it
                    if semaphore.name() != name {
                        continue;
                    }

                    // If since was passed and the semaphore was raised before since, ignore it
                    if semaphore.raised_at() < since {
                        continue;
                    }

                    return Ok(());
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(count)) => {
                    tracing::warn!(message = "synchronization service lagged", count = count);
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return Err(super::Error::Closed);
                }
            }
        }
    }

    /// Stops the synchronization service and its agent.
    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn stop(&self) -> Result<(), super::Error> {
        if let Some(handle) = {
            let mut guard = self.handle.write().await;
            guard.take()
        } {
            handle.stop().await?;
        }

        Ok(())
    }

    /// Raises a semaphore and notifies all waiters.
    #[instrument(level = "info", skip(self, tx))]
    pub(crate) async fn raise(
        &self,
        tx: &mut PgTransaction<'static>,
        name: &str,
    ) -> Result<semaphores::Semaphore, super::Error> {
        let semaphore = super::repositories::semaphores::queries::store(&mut **tx, name).await?;
        super::repositories::semaphores::queries::notify(&mut **tx, &semaphore).await?;
        Ok(semaphore)
    }
}

/// Listens for semaphore notifications and broadcasts them to waiters.
pub struct Agent {
    tx: broadcast::Sender<Semaphore>,
    rx: Option<oneshot::Receiver<()>>,
    semaphores: Arc<semaphores::Repository>,
    reference_time: DateTime<Utc>,
    stream: fx_pgmux::NotificationStream,
    poll_interval: Option<Duration>,
}

impl Agent {
    pub fn new(
        stream: fx_pgmux::NotificationStream,
        semaphores: &Arc<semaphores::Repository>,
        tx: &broadcast::Sender<Semaphore>,
        rx_stop: oneshot::Receiver<()>,
        poll_interval: &Option<Duration>,
    ) -> Self {
        Self {
            semaphores: semaphores.clone(),
            tx: tx.clone(),
            rx: Some(rx_stop),
            reference_time: Utc::now(),
            stream,
            poll_interval: poll_interval.clone(),
        }
    }

    /// Runs the agent loop, listening for notifications and polling.
    #[instrument(level = "debug", skip(self))]
    pub async fn run(&mut self) -> Result<(), super::Error> {
        let mut rx_stop = self
            .rx
            .take()
            .ok_or(super::Error::MissingTerminationChannel)?;

        let mut interval = {
            self.poll_interval.map(|duration| {
                let mut interval = tokio::time::interval(duration);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                interval.reset(); // ignore the very first poll
                interval
            })
        };

        loop {
            let poll_future = match interval {
                Some(ref mut interval) => Either::Left(interval.tick()),
                None => Either::Right(pending::<Instant>()),
            };

            tokio::select! {
                Some(notification) = self.stream.next() => {
                    let semaphore: semaphores::Semaphore = serde_json::from_str(&notification)?;
                    self.broadcast(semaphore)?;
                }
                // Conditionally await an interval tick if the configured duration is non-zero
                // Passing `Duration::ZERO` will effectively turn fallback polling off.
                _ = poll_future => {
                    let filter = semaphores::PollFilter::default();

                    tracing::warn!(message = "fallback polling triggered for synchronization service", reference_time=self.reference_time.to_rfc3339());
                    let semaphores = self.semaphores.poll(&self.reference_time, &filter).await?;

                    for semaphore in semaphores {
                        self.broadcast(semaphore)?;
                    }
                }
                _ = &mut rx_stop => {
                    tracing::warn!("Sync agent stopped");
                    break
                }
            };
        }

        Ok(())
    }

    fn broadcast(&mut self, semaphore: semaphores::Semaphore) -> Result<(), super::Error> {
        let raised_at = semaphore.raised_at().clone();

        // Move the reference time forward if reference_time is smaller than raised_at
        if self.reference_time < raised_at {
            self.reference_time = raised_at
        }

        // Broadcast
        self.tx.send(semaphore)?;

        Ok(())
    }
}

fn run_agent(mut agent: Agent) -> JoinHandle<Result<(), super::Error>> {
    tokio::spawn(async move {
        agent.run().await?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::PollIntervalSeconds;
    use crate::infrastructure::db;
    use futures::lock::Mutex;
    use sqlx::PgPool;
    use std::collections::HashSet;
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use tokio::task::JoinHandle;

    async fn setup(pool: PgPool) -> anyhow::Result<Arc<crate::services::synchronization::Service>> {
        let mut c = crate::infrastructure::di::Container::new();

        // Registrations
        crate::register(&mut c);

        // Overwrites
        c.provide(|_| Box::pin(async { Ok(PollIntervalSeconds { value: 60 }) }));
        let ro = pool.clone();
        c.provide(|_| Box::pin(async { Ok(db::ReadPool { pool: ro }) }));
        let wr = pool.clone();
        c.provide(|_| Box::pin(async { Ok(db::WritePool { pool: wr }) }));

        let svc = c
            .get::<Arc<crate::services::synchronization::Service>>()
            .await?;

        let mux = c.get::<Arc<Mutex<Option<Multiplexer>>>>().await?;

        let mut mux = mux.lock().await.take().expect("Could not take mux");

        svc.register(&mut mux).await?;

        tokio::spawn(mux.listen());

        Ok(svc)
    }

    #[sqlx::test(migrations = false)]
    async fn it_terminates_a_single_task(pool: PgPool) -> anyhow::Result<()> {
        crate::test_tools::init_test_tracing();
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = setup(pool.clone()).await?;

        let since = Utc::now();
        let (tx, mut rx) = mpsc::channel(1);
        wait_for("task-A", &svc, &tx, &since);

        {
            let mut tx = pool.begin().await?;
            svc.raise(&mut tx, "task-A").await?;
            tx.commit().await?;
        }

        let expected_length = 1;
        let mut actual = Vec::with_capacity(expected_length);
        while let Some(actual_name) = rx.recv().await {
            actual.push(actual_name);
            if actual.len() == expected_length {
                break;
            }
        }

        svc.stop().await?;

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_terminates_subset_of_tasks(pool: PgPool) -> anyhow::Result<()> {
        crate::test_tools::init_test_tracing();
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = setup(pool.clone()).await?;

        let since = Utc::now();
        let (tx, mut rx) = mpsc::channel(10);
        let _ = wait_for("task-A", &svc, &tx, &since);
        let b = wait_for("task-B", &svc, &tx, &since);

        {
            let mut tx = pool.begin().await?;
            svc.raise(&mut tx, "task-A").await?;
            tx.commit().await?;
        }

        let expected_length = 1;
        let mut actual = Vec::with_capacity(expected_length);
        while let Some(actual_name) = rx.recv().await {
            actual.push(actual_name);
            if actual.len() == expected_length {
                // once we got what we expected to get, abort task b
                // we need to do this because the test helper is spawning a task that will otherwise
                // continue waiting for `read.wait_for(name, &since).await?;` so resolve
                b.abort();
                break;
            }
        }

        svc.stop().await?;

        assert_eq!(vec!["task-A"], actual);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_resolves_all_waiting_tasks(pool: PgPool) -> anyhow::Result<()> {
        crate::test_tools::init_test_tracing();
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = setup(pool.clone()).await?;

        let since = Utc::now();
        let (tx, mut rx) = mpsc::channel(4);
        wait_for("comp-A", &svc, &tx, &since);
        wait_for("comp-B", &svc, &tx, &since);
        wait_for("comp-C", &svc, &tx, &since);
        wait_for("comp-D", &svc, &tx, &since);

        {
            let mut tx = pool.begin().await?;
            svc.raise(&mut tx, "comp-B").await?;
            tx.commit().await?;

            let mut tx = pool.begin().await?;
            svc.raise(&mut tx, "comp-D").await?;
            tx.commit().await?;

            let mut tx = pool.begin().await?;
            svc.raise(&mut tx, "comp-A").await?;
            tx.commit().await?;

            let mut tx = pool.begin().await?;
            svc.raise(&mut tx, "comp-C").await?;
            tx.commit().await?;
        }

        let expected_length = 4;
        let mut actual = Vec::<&str>::with_capacity(expected_length);

        while let Some(name) = rx.recv().await {
            actual.push(name);

            if actual.len() == expected_length {
                break;
            }
        }

        svc.stop().await?;

        let expected: HashSet<&str> = HashSet::from(["comp-A", "comp-B", "comp-C", "comp-D"]);
        let actual: HashSet<&str> = actual.into_iter().collect();
        assert_eq!(expected, actual);

        Ok(())
    }

    fn wait_for(
        name: &'static str,
        svc: &Arc<Service>,
        tx: &mpsc::Sender<&'static str>,
        since: &DateTime<Utc>,
    ) -> JoinHandle<Result<(), anyhow::Error>> {
        let since = since.clone();
        let svc = svc.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            svc.wait_for(name, &since).await?;
            tx.send(name).await?;
            Ok(())
        })
    }
}
