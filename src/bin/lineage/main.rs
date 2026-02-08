use askama::Template;
use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Redirect, Response},
    routing::get,
};
use fx_durable_ga::{
    bootstrap,
    models::{Evaluation, Genotype},
    services::genotype_explorer,
};
use sqlx::postgres::PgPoolOptions;
use std::{collections::HashMap, sync::Arc};
use tracing::{Level, instrument};
use uuid::Uuid;
const DEFAULT_DEGREE: u32 = 3;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::from_filename(".env.local").ok();
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    let svc_builder = bootstrap::ApplicationBuilder::default().with_pool(pool);
    let svc_explorer = Arc::new(svc_builder.build_explorer_svc());

    let router = Router::new()
        .route("/lineage", get(lineage_get))
        .with_state(ExplorerState {
            svc: Arc::clone(&svc_explorer),
        });

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, router).await?;
    Ok(())
}

fn build_descendant_lookup(
    nodes: &HashMap<Uuid, (Genotype, Evaluation)>,
) -> HashMap<Uuid, Vec<Uuid>> {
    let mut descendants: HashMap<Uuid, Vec<Uuid>> = HashMap::new();

    for (genotype, _) in nodes.values() {
        let id = genotype.id();

        if let Some(parent_id) = genotype.parent_a() {
            if nodes.contains_key(parent_id) {
                descendants.entry(*parent_id).or_default().push(id);
            }
        }

        if let Some(parent_id) = genotype.parent_b() {
            if nodes.contains_key(parent_id) {
                descendants.entry(*parent_id).or_default().push(id);
            }
        }
    }

    descendants
}

fn build_descendant_tree(
    genotype_id: &Uuid,
    nodes: &HashMap<Uuid, (Genotype, Evaluation)>,
    descendants: &HashMap<Uuid, Vec<Uuid>>,
    degree: u32,
    direction: Direction,
) -> serde_json::Value {
    let (genotype, evaluation) = match nodes.get(genotype_id) {
        Some(pair) => pair,
        None => return serde_json::Value::Null,
    };

    let mut children = Vec::new();
    if let Some(child_ids) = descendants.get(genotype_id) {
        for child_id in child_ids {
            if nodes.contains_key(child_id) {
                children.push(build_descendant_tree(
                    child_id,
                    nodes,
                    descendants,
                    degree,
                    direction,
                ));
            }
        }
    }

    let href = LineageQuery::new(genotype_id, degree, direction).url();

    serde_json::json!({
        "name": format!("{:.3}", evaluation.fitness()),
        "id": genotype.id().to_string(),
        "fitness": evaluation.fitness(),
        "label": genotype.type_name(),
        "href": href,
        "children": children,
    })
}

fn build_lineage_json(
    genotype_id: Option<&Uuid>,
    records: &[(Genotype, Evaluation)],
    degree: u32,
    direction: Direction,
) -> Option<String> {
    let root_id = match genotype_id {
        Some(id) if !records.is_empty() => id,
        _ => return None,
    };

    let nodes: HashMap<Uuid, (Genotype, Evaluation)> = records
        .iter()
        .map(|(genotype, evaluation)| (genotype.id(), (genotype.clone(), evaluation.clone())))
        .collect();

    let tree = match direction {
        Direction::Ancestors => build_ancestor_tree(root_id, &nodes, degree, direction),
        Direction::Descendants => {
            let descendants = build_descendant_lookup(&nodes);
            build_descendant_tree(root_id, &nodes, &descendants, degree, direction)
        }
    };

    match serde_json::to_string(&tree) {
        Ok(value) => Some(value),
        Err(err) => {
            tracing::error!(error = %err, "failed to serialize ancestor tree to JSON");
            None
        }
    }
}

fn build_ancestor_tree(
    genotype_id: &Uuid,
    nodes: &HashMap<Uuid, (Genotype, Evaluation)>,
    degree: u32,
    direction: Direction,
) -> serde_json::Value {
    let (genotype, evaluation) = match nodes.get(genotype_id) {
        Some(pair) => pair,
        None => return serde_json::Value::Null,
    };

    let mut children = Vec::new();
    if let Some(parent_id) = genotype.parent_a() {
        if nodes.contains_key(parent_id) {
            children.push(build_ancestor_tree(parent_id, nodes, degree, direction));
        }
    }
    if let Some(parent_id) = genotype.parent_b() {
        if nodes.contains_key(parent_id) {
            children.push(build_ancestor_tree(parent_id, nodes, degree, direction));
        }
    }

    let href = LineageQuery::new(genotype_id, degree, direction).url();

    serde_json::json!({
        "name": format!("{:.3}", evaluation.fitness()),
        "id": genotype.id().to_string(),
        "fitness": evaluation.fitness(),
        "label": genotype.type_name(),
        "href": href,
        "children": children,
    })
}

#[derive(Clone)]
struct ExplorerState {
    svc: Arc<genotype_explorer::Service>,
}

#[derive(Debug, Template)]
#[template(path = "lineage.html")]
struct LineageTemplate {
    genotype_id: Option<Uuid>,
    degree: u32,
    direction: String,
    lineage_json: Option<String>,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum Direction {
    Ancestors,
    Descendants,
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ancestors => write!(f, "ancestors"),
            Self::Descendants => write!(f, "descendants"),
        }
    }
}

impl Default for Direction {
    fn default() -> Self {
        Self::Ancestors
    }
}

#[derive(Debug, Default, serde::Deserialize)]
struct LineageQuery {
    #[serde(default)]
    genotype_id: Option<Uuid>,
    #[serde(default)]
    degree: Option<u32>,
    #[serde(default)]
    direction: Option<Direction>,
}

impl LineageQuery {
    fn new(genotype_id: &Uuid, degree: u32, direction: Direction) -> Self {
        Self {
            genotype_id: Some(*genotype_id),
            degree: Some(degree),
            direction: Some(direction),
        }
    }

    fn url(&self) -> String {
        let degree = self.degree.unwrap_or(DEFAULT_DEGREE);

        let direction = self.direction.unwrap_or(Direction::default());

        let mut url = format!("/lineage?degree={}&direction={}", degree, direction);

        if let Some(genotype_id) = self.genotype_id {
            url.push_str(&format!("&genotype_id={}", genotype_id));
        }

        return url;
    }

    fn should_redirect(&self) -> bool {
        self.degree.is_none() || self.direction.is_none()
    }
}

#[instrument(
    level = "debug",
    skip(state, params),
    fields(genotype_id = ?params.genotype_id, degree = ?params.degree, direction = ?params.direction)
)]
async fn lineage_get(
    State(state): State<ExplorerState>,
    Query(params): Query<LineageQuery>,
) -> Result<LineageTemplate, Redirect> {
    let needs_redirect = params.should_redirect();
    let degree = params.degree.unwrap_or(DEFAULT_DEGREE);
    let direction = params.direction.unwrap_or(Direction::default());

    if needs_redirect {
        let redirect_query = LineageQuery {
            genotype_id: params.genotype_id,
            degree: Some(degree),
            direction: Some(direction),
        };
        return Err(Redirect::permanent(&redirect_query.url()));
    }

    let genotype_id = params.genotype_id;

    let lineage_records = match genotype_id.as_ref() {
        Some(genotype_id) => {
            let result = match direction {
                Direction::Ancestors => state.svc.get_ancestors(genotype_id, degree).await,
                Direction::Descendants => state.svc.get_descendants(genotype_id, degree).await,
            };

            match result {
                Ok(records) => records,
                Err(err) => {
                    tracing::error!(error = %err, "failed to fetch lineage records");
                    Vec::new()
                }
            }
        }
        None => Vec::new(),
    };

    let lineage_json =
        build_lineage_json(genotype_id.as_ref(), &lineage_records, degree, direction);

    Ok(LineageTemplate {
        genotype_id,
        degree,
        direction: direction.to_string(),
        lineage_json,
    })
}

impl IntoResponse for LineageTemplate {
    fn into_response(self) -> Response {
        match self.render() {
            Ok(body) => Html(body).into_response(),
            Err(err) => {
                tracing::error!(error = %err, "failed to render lineage template");
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        }
    }
}
