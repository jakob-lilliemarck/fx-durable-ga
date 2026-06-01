use futures::future::BoxFuture;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use tracing::instrument;

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("No provider registered for {type_name} with id {type_id:?}")]
    NoProvider {
        type_name: &'static str,
        type_id: TypeId,
    },

    #[error("Cycle detected:\n{0}")]
    CycleDetected(Cycle),

    #[error("No value for: {type_name}")]
    NoValue { type_name: &'static str },

    #[error("Could not provide: {type_name}: {error}")]
    ProvideError {
        type_name: &'static str,
        #[source]
        error: Box<dyn std::error::Error + Sync + Send>,
    },
}

impl ProviderError {
    fn cycle(entries: &[ResolutionEntry], recurring_type: &'static str) -> Self {
        Self::CycleDetected(Cycle {
            entries: entries.to_vec(),
            recurring_type,
        })
    }

    pub fn new<T, E>(error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::ProvideError {
            type_name: std::any::type_name::<T>(),
            error: Box::new(error),
        }
    }
}

pub type ProviderResult<T> = Result<T, ProviderError>;

pub struct Cycle {
    entries: Vec<ResolutionEntry>,
    recurring_type: &'static str,
}

impl std::fmt::Display for Cycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let culprit = self.entries.last().map(|e| &e.type_name);
        for entry in self.entries.iter() {
            if entry.type_name == self.recurring_type {
                writeln!(f, "\t{} ← first occurrence", entry.type_name)?;
            } else if Some(&entry.type_name) == culprit {
                writeln!(
                    f,
                    "\t{} ← provider depends on {}",
                    entry.type_name, self.recurring_type
                )?;
            } else {
                writeln!(f, "\t{}", entry.type_name)?;
            }
        }
        writeln!(f, "\t{}", self.recurring_type)
    }
}

impl std::fmt::Debug for Cycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum InvokeError {
    #[error("provide error")]
    Provide(#[from] ProviderError),

    #[error("Could not invoke: {error}")]
    Invoke {
        #[source]
        error: Box<dyn std::error::Error + Sync + Send>,
    },
}

impl InvokeError {
    pub fn new<E>(error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Invoke {
            error: Box::new(error),
        }
    }
}

pub type InvokeResult = Result<(), InvokeError>;
// ── Provider type ─────────────────────────────────────────────────────────────

type Provider = Box<
    dyn for<'a> FnOnce(
            &'a mut Container,
        ) -> BoxFuture<'a, Result<Box<dyn Any + Send + Sync>, ProviderError>>
        + Send
        + Sync,
>;

// ── Invokeable type ─────────────────────────────────────────────────────────────
type Invokeable =
    Box<dyn for<'a> FnOnce(&'a mut Container) -> BoxFuture<'a, InvokeResult> + Send + Sync>;

// ── Container ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ResolutionEntry {
    type_id: TypeId,
    type_name: &'static str,
}

pub struct Container {
    values: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    providers: HashMap<TypeId, Provider>,
    invokables: Vec<Invokeable>,
    resolution_path: Vec<ResolutionEntry>,
}

impl Container {
    #[instrument(level = "debug", skip_all)]
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            providers: HashMap::new(),
            invokables: Vec::new(),
            resolution_path: Vec::new(),
        }
    }

    /// Register an async provider for type `T`. Can depend on other registered
    /// types via `builder.get::<Dep>().await?` inside the closure.
    #[instrument(level = "debug", skip_all, fields(type_name))]
    pub fn provide<T, F>(&mut self, f: F)
    where
        T: Any + Send + Sync + Clone,
        F: for<'a> FnOnce(&'a mut Container) -> BoxFuture<'a, ProviderResult<T>>
            + Send
            + Sync
            + 'static,
    {
        let type_name = std::any::type_name::<T>();
        tracing::Span::current().record("type_name", type_name);
        tracing::debug!("di::Container::provide");

        self.providers.insert(
            TypeId::of::<T>(),
            Box::new(move |c| {
                Box::pin(async move {
                    f(c).await
                        .map(|v| Box::new(v) as Box<dyn Any + Send + Sync>)
                }) as BoxFuture<'_, _>
            }),
        );
    }

    /// Get a type from the container
    #[instrument(level = "debug", skip(self), fields(type_name))]
    pub async fn get<T: Any + Send + Sync + Clone>(&mut self) -> Result<T, ProviderError> {
        let type_id = TypeId::of::<T>();
        let type_name = std::any::type_name::<T>();

        tracing::Span::current().record("type_name", type_name);
        tracing::debug!(
            message = "di::Container::get",
            resolution_path = ?self.resolution_path
        );

        // guard against cycles
        if self
            .resolution_path
            .iter()
            .any(|entry| entry.type_id == type_id)
        {
            return Err(ProviderError::cycle(&self.resolution_path, type_name));
        }

        // push the current type to the resolution path
        self.resolution_path
            .push(ResolutionEntry { type_id, type_name });

        // try to resolve the type
        let result = self.resolve(type_id).await;

        // pop from the resolution stack
        self.resolution_path.pop();

        result
    }

    /// Resolve a type, calling its provider if it has not yet been constructed.
    #[instrument(level = "debug", skip(self), fields(type_name))]
    async fn resolve<T: Any + Send + Sync + Clone>(
        &mut self,
        type_id: TypeId,
    ) -> Result<T, ProviderError> {
        let type_id = TypeId::of::<T>();
        let type_name = std::any::type_name::<T>();

        if !self.values.contains_key(&type_id) {
            let provider = self
                .providers
                .remove(&type_id)
                .ok_or_else(|| ProviderError::NoProvider { type_name, type_id })?;

            let value = provider(self).await?;

            self.values.insert(type_id, value);
        }

        self.values
            .get(&type_id)
            .and_then(|v| v.downcast_ref::<T>())
            .cloned()
            .ok_or_else(|| ProviderError::NoValue { type_name })
    }

    #[instrument(level = "debug", skip_all)]
    pub fn invokable<F>(&mut self, f: F)
    where
        F: for<'a> FnOnce(&'a mut Container) -> BoxFuture<'a, InvokeResult> + Send + Sync + 'static,
    {
        tracing::debug!(message = "di::Container::invokable");

        self.invokables.push(Box::new(f))
    }

    #[instrument(level = "debug", skip_all)]
    pub async fn invoke(&mut self) -> InvokeResult {
        tracing::debug!(message = "di::Container::invoke");

        let invokables = std::mem::take(&mut self.invokables);
        for invokable in invokables {
            invokable(self).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use futures::future::BoxFuture;
    use std::sync::Arc;

    #[derive(Clone)]
    pub struct A;

    #[derive(Clone)]
    pub struct B {
        _a: A,
    }

    #[derive(Clone)]
    pub struct C {
        _a: A,
        _b: Arc<B>,
    }

    fn provide_a(_: &mut super::Container) -> BoxFuture<'_, super::ProviderResult<A>> {
        Box::pin(async { Ok(A) })
    }

    fn provide_b(container: &mut super::Container) -> BoxFuture<'_, super::ProviderResult<Arc<B>>> {
        Box::pin(async {
            let a = container.get::<A>().await?;
            Ok(Arc::new(B { _a: a }))
        })
    }

    fn provide_c(container: &mut super::Container) -> BoxFuture<'_, super::ProviderResult<Arc<C>>> {
        Box::pin(async {
            let a = container.get::<A>().await?;
            let b = container.get::<Arc<B>>().await?;
            Ok(Arc::new(C { _a: a, _b: b }))
        })
    }

    #[tokio::test]
    async fn it_constructs_a_dag() -> anyhow::Result<()> {
        let mut container = super::Container::new();

        container.provide(provide_a);
        container.provide(provide_b);
        container.provide(provide_c);

        container.get::<Arc<C>>().await?;

        Ok(())
    }

    #[derive(Clone)]
    pub struct D;

    #[derive(Clone)]
    pub struct E;

    #[derive(Clone)]
    pub struct F;

    fn provide_d(c: &mut super::Container) -> BoxFuture<'_, super::ProviderResult<D>> {
        Box::pin(async {
            c.get::<E>().await?;
            Ok(D)
        })
    }

    fn provide_e(c: &mut super::Container) -> BoxFuture<'_, super::ProviderResult<E>> {
        Box::pin(async {
            c.get::<F>().await?;
            Ok(E)
        })
    }

    fn provide_f(c: &mut super::Container) -> BoxFuture<'_, super::ProviderResult<F>> {
        Box::pin(async {
            c.get::<D>().await?;
            Ok(F)
        })
    }

    #[tokio::test]
    async fn it_detects_cycles_and_provides_friendly_errors() -> anyhow::Result<()> {
        crate::test_tools::init_test_tracing();

        let mut container = super::Container::new();

        container.provide::<D, _>(provide_d);
        container.provide::<E, _>(provide_e);
        container.provide::<F, _>(provide_f);

        let result = container.get::<E>().await;

        assert!(result.is_err(), "expected Err, got Ok");

        if let Err(err) = result {
            println!("{}", err)
        }

        Ok(())
    }
}
