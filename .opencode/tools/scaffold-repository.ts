import { tool } from "@opencode-ai/plugin";
import * as fs from "fs";
import * as path from "path";

export default tool({
    description:
    "Scaffold a repository aggregate under an existing service. Creates boilerplate: mod.rs, errors.rs, repository.rs, registrations.rs, models.rs, and an empty queries/mod.rs. Optionally updates the parent repositories/mod.rs and service mod.rs with re-exports.",
  args: {
    serviceName: tool.schema
      .string()
      .describe("Directory under src/services/ (e.g., evaluation)"),
    aggregateName: tool.schema
      .string()
      .describe(
        "Plural underscore-separated directory name (e.g., evaluations). Same concept as singularName, different case.",
      ),
    singularName: tool.schema
      .string()
      .describe(
        "PascalCase domain model struct name (e.g., Evaluation). Same concept as aggregateName, different case.",
      ),
    updateModFiles: tool.schema
      .boolean()
      .optional()
      .default(true)
      .describe(
        "Update parent repositories/mod.rs and service mod.rs (default: true)",
      ),
  },
  async execute(args, context) {
    const { serviceName, aggregateName, singularName, updateModFiles } = args;
    const root = context.worktree;

    const repoDir = path.join(
      root,
      "src",
      "services",
      serviceName,
      "repositories",
      aggregateName,
    );
    const queriesDir = path.join(repoDir, "queries");
    fs.mkdirSync(queriesDir, { recursive: true });

    writeFile(path.join(repoDir, "errors.rs"), errorsContent());
    writeFile(path.join(repoDir, "models.rs"), modelsContent(singularName));
    writeFile(
      path.join(repoDir, "registrations.rs"),
      registrationsContent(aggregateName),
    );
    writeFile(path.join(repoDir, "mod.rs"), modContent(singularName));
    writeFile(
      path.join(repoDir, "repository.rs"),
      repositoryContent(singularName),
    );
    writeFile(path.join(queriesDir, "mod.rs"), queriesModContent());

    const messages: string[] = [`Created ${path.relative(root, repoDir)}/`];

    if (updateModFiles) {
      messages.push(updateRepositoriesMod(root, serviceName, aggregateName));
      messages.push(
        updateServiceMod(root, serviceName, aggregateName, singularName),
      );
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
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
`;
}

function modelsContent(singular: string): string {
  return `use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ${singular} {
    pub(crate) id: Uuid,
}

impl ${singular} {
    pub fn new() -> Self {
        Self { id: Uuid::now_v7() }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }
}

impl Default for ${singular} {
    fn default() -> Self {
        Self::new()
    }
}
`;
}

function registrationsContent(name: string): string {
  return `use crate::infrastructure::db;
use crate::infrastructure::di::{Container, ProviderResult};
use futures::future::BoxFuture;

pub fn provide_${name}_repository_ro(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Read>> {
    Box::pin(async {
        let ro = c.get::<db::ReadPool>().await?;
        Ok(super::Read::new(ro))
    })
}

pub fn provide_${name}_repository_wr(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Write>> {
    Box::pin(async {
        let wr = c.get::<db::WritePool>().await?;
        Ok(super::Write::new(wr))
    })
}
`;
}

function modContent(singular: string): string {
  return `mod errors;
mod models;
pub(super) mod queries;
mod repository;

pub mod registrations;

pub use errors::Error;
pub use models::${singular};

pub use repository::Read;
pub(crate) use repository::Write;
pub(crate) use repository::WriteTx;
`;
}

function repositoryContent(singular: string): string {
  return `use super::errors::Error;
use super::models::${singular};
use crate::infrastructure::db;
use sqlx::PgTransaction;

#[derive(Debug, Clone)]
pub struct Read {
    ro: db::ReadPool,
}

#[derive(Debug, Clone)]
pub struct Write {
    wr: db::WritePool,
}

pub struct WriteTx<'tx> {
    tx: &'tx mut PgTransaction<'static>,
}

impl db::Tx for Write {
    type Error = Error;

    fn tx(self) -> db::TxFut<Self::Error> {
        let pool = self.wr.pool.clone();
        Box::pin(async move {
            let tx = pool.begin().await?;
            Ok(tx)
        })
    }
}

impl Read {
    pub fn new(ro: db::ReadPool) -> Self {
        Self { ro }
    }
}

impl Write {
    pub fn new(wr: db::WritePool) -> Self {
        Self { wr }
    }
}

impl<'tx> WriteTx<'tx> {
    pub fn new(tx: &'tx mut PgTransaction<'static>) -> Self {
        Self { tx }
    }
}
`;
}

function queriesModContent(): string {
  return `// Query functions will be added per repository concern.
// See AGENTS.md for query conventions and the create-repository skill for guidance.
`;
}

function readOptionalFile(p: string): string | null {
  try {
    return fs.readFileSync(p, "utf-8");
  } catch {
    return null;
  }
}

function updateRepositoriesMod(
  root: string,
  service: string,
  aggregate: string,
): string {
  const parentDir = path.join(root, "src", "services", service, "repositories");
  const modPath = path.join(parentDir, "mod.rs");
  const line = `pub(super) mod ${aggregate};\n`;

  const existing = readOptionalFile(modPath);

  if (existing !== null) {
    if (existing.includes(aggregate)) {
      return `${path.relative(root, modPath)}: already has module ${aggregate} (skipped)`;
    }
    fs.writeFileSync(modPath, existing + line, "utf-8");
  } else {
    fs.mkdirSync(parentDir, { recursive: true });
    fs.writeFileSync(modPath, line, "utf-8");
  }

  return `Updated ${path.relative(root, modPath)}`;
}

function updateServiceMod(
  root: string,
  service: string,
  aggregate: string,
  singular: string,
): string {
  const modPath = path.join(root, "src", "services", service, "mod.rs");
  const existing = readOptionalFile(modPath);

  if (existing === null) {
    return `${path.relative(root, modPath)}: not found (skipped)`;
  }

  const reexportLine = `pub use repositories::${aggregate}::${singular};\n`;
  const readLine = `pub use repositories::${aggregate}::Read as ${singular}Read;\n`;

  if (existing.includes(reexportLine.trim())) {
    return `${path.relative(root, modPath)}: already has re-exports for ${singular} (skipped)`;
  }

  const lines = existing.split("\n");

  // Find the best insertion point: after the last pub use line in the pub use block
  let insertIdx = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    const trimmed = lines[i].trim();
    if (
      trimmed.startsWith("pub use ") ||
      trimmed.startsWith("pub(crate) use ")
    ) {
      insertIdx = i + 1;
      break;
    }
  }
  // Fallback: after module declarations
  if (insertIdx === -1) {
    for (let i = 0; i < lines.length; i++) {
      if (
        !lines[i].trim().startsWith("mod ") &&
        !lines[i].trim().startsWith("//")
      ) {
        insertIdx = i;
        break;
      }
    }
  }
  if (insertIdx === -1) {
    insertIdx = lines.length;
  }

  lines.splice(insertIdx, 0, reexportLine.trimEnd(), readLine.trimEnd());
  fs.writeFileSync(modPath, lines.join("\n"), "utf-8");

  return `Updated ${path.relative(root, modPath)}`;
}
