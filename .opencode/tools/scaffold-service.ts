import { tool } from "@opencode-ai/plugin";
import * as fs from "fs";
import * as path from "path";

export default tool({
  description:
    "Scaffold a new service module under src/services/. Creates the required boilerplate (mod.rs, errors.rs, service.rs, registrations.rs) and optionally events.rs, jobs.rs, and repositories/mod.rs. Optionally updates src/services/mod.rs with the module declaration.",
  args: {
    serviceName: tool.schema
      .string()
      .describe(
        "Directory under src/services/ (e.g., genotype_indexing)",
      ),
    withEvents: tool.schema
      .boolean()
      .optional()
      .default(false)
      .describe("Generate events.rs stub (default: false)"),
    withJobs: tool.schema
      .boolean()
      .optional()
      .default(false)
      .describe("Generate jobs.rs stub (default: false)"),
    withRepositories: tool.schema
      .boolean()
      .optional()
      .default(false)
      .describe("Generate repositories/mod.rs stub (default: false)"),
    updateParentMod: tool.schema
      .boolean()
      .optional()
      .default(true)
      .describe("Update src/services/mod.rs (default: true)"),
  },
  async execute(args, context) {
    const { serviceName, withEvents, withJobs, withRepositories, updateParentMod } = args;
    const root = context.worktree;

    const serviceDir = path.join(root, "src", "services", serviceName);

    fs.mkdirSync(serviceDir, { recursive: true });

    if (withRepositories) {
      fs.mkdirSync(path.join(serviceDir, "repositories"), { recursive: true });
    }

    writeFile(path.join(serviceDir, "errors.rs"), errorsContent());
    writeFile(path.join(serviceDir, "service.rs"), serviceContent(serviceName));
    writeFile(
      path.join(serviceDir, "registrations.rs"),
      registrationsContent(serviceName, withEvents, withJobs),
    );
    writeFile(
      path.join(serviceDir, "mod.rs"),
      modContent(serviceName, withEvents, withJobs, withRepositories),
    );

    if (withEvents) {
      writeFile(path.join(serviceDir, "events.rs"), eventsContent());
    }

    if (withJobs) {
      writeFile(path.join(serviceDir, "jobs.rs"), jobsContent());
    }

    if (withRepositories) {
      writeFile(path.join(serviceDir, "repositories", "mod.rs"), repositoriesModContent());
    }

    const messages: string[] = [
      `Created ${path.relative(root, serviceDir)}/`,
    ];

    if (updateParentMod) {
      messages.push(updateServicesMod(root, serviceName));
    }

    return messages.join("\n");
  },
});

function writeFile(p: string, content: string) {
  fs.writeFileSync(p, content, "utf-8");
}

function errorsContent(): string {
  return `#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Catch-all for errors that don't map to a domain variant.
    /// Omit this if the service doesn't use db::begin.
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
`;
}

function serviceContent(name: string): string {
  return `use tracing::instrument;

pub struct Service {
}

impl Service {
    pub(crate) fn new() -> Self {
        Self { }
    }

    /// Placeholder — replace with actual business methods.
    #[instrument(level = "debug", skip(self))]
    pub async fn do_something(&self) -> Result<(), super::Error> {
        Ok(())
    }
}
`;
}

function registrationsContent(
  name: string,
  withEvents: boolean,
  withJobs: boolean,
): string {
  let commentBlock = "";
  if (withEvents && withJobs) {
    commentBlock = `
    // c.invokable(invoke_event_handler_registration);
    // c.invokable(invoke_job_handler_registration);`;
  } else if (withEvents) {
    commentBlock = `
    // c.invokable(invoke_event_handler_registration);`;
  } else if (withJobs) {
    commentBlock = `
    // c.invokable(invoke_job_handler_registration);`;
  }

  return `use crate::infrastructure::di::{Container, ProviderResult};
use futures::future::BoxFuture;
use std::sync::Arc;

fn provide_${name}_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        Ok(Arc::new(super::Service::new()))
    })
}

pub fn register(c: &mut Container) {
    c.provide(provide_${name}_service);${commentBlock}
}
`;
}

function modContent(
  name: string,
  withEvents: boolean,
  withJobs: boolean,
  withRepositories: boolean,
): string {
  const lines: string[] = [];

  lines.push("mod errors;");

  if (withEvents) {
    lines.push("mod events;");
  } else {
    lines.push("// mod events;     // uncomment if events.rs exists");
  }

  if (withJobs) {
    lines.push("mod jobs;");
  } else {
    lines.push("// mod jobs;       // uncomment if jobs.rs exists");
  }

  lines.push("mod registrations;");
  lines.push("mod service;");

  if (withRepositories) {
    lines.push("pub(super) mod repositories;");
  } else {
    lines.push("// pub(super) mod repositories;  // uncomment if repositories/ exists");
  }

  lines.push("");
  lines.push("pub use errors::Error;");
  lines.push("pub use registrations::register;");
  lines.push("pub use service::Service;");
  lines.push("");
  lines.push("// Re-export event payloads");
  lines.push("// pub use events::SomethingHappenedEvent;");
  lines.push("");
  lines.push("// Re-export read repositories (never Write/WriteTx)");
  lines.push("// pub use repositories::<aggregate>::Read as <Singular>Read;");
  lines.push("// pub use repositories::<aggregate>::<Singular>;");
  lines.push("// pub use repositories::<aggregate>::<Filter>;");

  return lines.join("\n") + "\n";
}

function eventsContent(): string {
  return `use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomethingHappenedEvent {
    pub entity_id: Uuid,
}

impl fx_event_bus::Event for SomethingHappenedEvent {
    const NAME: &'static str = "SomethingHappened";
}

impl SomethingHappenedEvent {
    pub fn new(entity_id: Uuid) -> Self {
        Self { entity_id }
    }
}
`;
}

function jobsContent(): string {
  return `use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct DoSomethingMessage {
    pub entity_id: Uuid,
}

impl fx_mq_jobs::Message for DoSomethingMessage {
    const NAME: &str = "DoSomething";
}

pub(super) struct DoSomethingHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for DoSomethingHandler {
    type Message = DoSomethingMessage;
    type Error = super::Error;

    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            let _ = message;
            Ok(())
        })
    }

    fn max_attempts(&self) -> i32 { 5 }
}
`;
}

function repositoriesModContent(): string {
  return `// Repository aggregates will be scaffolded with the scaffold-repository tool.
`;
}

function readOptionalFile(p: string): string | null {
  try {
    return fs.readFileSync(p, "utf-8");
  } catch {
    return null;
  }
}

function updateServicesMod(root: string, name: string): string {
  const modPath = path.join(root, "src", "services", "mod.rs");
  const line = `pub mod ${name};\n`;

  const existing = readOptionalFile(modPath);

  if (existing !== null) {
    if (existing.includes(name)) {
      return `${path.relative(root, modPath)}: already has module ${name} (skipped)`;
    }
    fs.writeFileSync(modPath, existing + line, "utf-8");
  } else {
    fs.mkdirSync(path.dirname(modPath), { recursive: true });
    fs.writeFileSync(modPath, line, "utf-8");
  }

  return `Updated ${path.relative(root, modPath)}`;
}
