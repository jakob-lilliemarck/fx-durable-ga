use crate::bootstrap::App;
use crate::repositories::genotypes::Genotype;
use aide::axum::ApiRouter;
use askama::Template;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Redirect, Response},
    routing::get,
};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;
use uuid::Uuid;

const DEFAULT_DEGREE: u32 = 3;

pub fn router(app: Arc<App>) -> ApiRouter {
    ApiRouter::new()
        .route("/lineage", get(lineage_get))
        .with_state(app)
}

fn build_descendant_lookup(nodes: &HashMap<Uuid, Genotype>) -> HashMap<Uuid, Vec<Uuid>> {
    let mut descendants: HashMap<Uuid, Vec<Uuid>> = HashMap::new();

    for genotype in nodes.values() {
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
    nodes: &HashMap<Uuid, Genotype>,
    evaluations: &HashMap<Uuid, f64>,
    descendants: &HashMap<Uuid, Vec<Uuid>>,
    degree: u32,
    direction: Direction,
) -> serde_json::Value {
    let genotype = match nodes.get(genotype_id) {
        Some(genotype) => genotype,
        None => return serde_json::Value::Null,
    };
    let fitness = match evaluations.get(genotype_id) {
        Some(f) => *f,
        None => return serde_json::Value::Null,
    };

    let mut children = Vec::new();
    if let Some(child_ids) = descendants.get(genotype_id) {
        let mut ordered_child_ids = child_ids.clone();
        ordered_child_ids.sort_by_key(|id| id.to_string());
        for child_id in ordered_child_ids {
            if nodes.contains_key(&child_id) {
                children.push(build_descendant_tree(
                    &child_id,
                    nodes,
                    evaluations,
                    descendants,
                    degree,
                    direction,
                ));
            }
        }
    }

    let href = LineageQuery::new(genotype_id, degree, direction).url();

    serde_json::json!({
        "name": format!("{:.3}", fitness),
        "id": genotype.id().to_string(),
        "fitness": fitness,
        "label": genotype.type_name(),
        "href": href,
        "children": children,
    })
}

fn build_lineage_json(
    genotype_id: Option<&Uuid>,
    records: &[Genotype],
    evaluations: &HashMap<Uuid, f64>,
    degree: u32,
    direction: Direction,
) -> Option<String> {
    let root_id = match genotype_id {
        Some(id) if !records.is_empty() => id,
        _ => return None,
    };

    let nodes: HashMap<Uuid, Genotype> = records
        .iter()
        .map(|genotype| (genotype.id(), genotype.clone()))
        .collect();

    let tree = match direction {
        Direction::Ancestors => {
            build_ancestor_tree(root_id, &nodes, evaluations, degree, direction)
        }
        Direction::Descendants => {
            let descendants = build_descendant_lookup(&nodes);
            build_descendant_tree(
                root_id,
                &nodes,
                evaluations,
                &descendants,
                degree,
                direction,
            )
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
    nodes: &HashMap<Uuid, Genotype>,
    evaluations: &HashMap<Uuid, f64>,
    degree: u32,
    direction: Direction,
) -> serde_json::Value {
    let genotype = match nodes.get(genotype_id) {
        Some(genotype) => genotype,
        None => return serde_json::Value::Null,
    };
    let fitness = match evaluations.get(genotype_id) {
        Some(f) => *f,
        None => return serde_json::Value::Null,
    };

    let mut children = Vec::new();
    let mut parent_ids = Vec::new();
    if let Some(parent_id) = genotype.parent_a() {
        if nodes.contains_key(parent_id) {
            parent_ids.push(*parent_id);
        }
    }
    if let Some(parent_id) = genotype.parent_b() {
        if nodes.contains_key(parent_id) {
            parent_ids.push(*parent_id);
        }
    }
    parent_ids.sort_by_key(|id| id.to_string());
    for parent_id in parent_ids {
        children.push(build_ancestor_tree(
            &parent_id,
            nodes,
            evaluations,
            degree,
            direction,
        ));
    }

    let href = LineageQuery::new(genotype_id, degree, direction).url();

    serde_json::json!({
        "name": format!("{:.3}", fitness),
        "id": genotype.id().to_string(),
        "fitness": fitness,
        "label": genotype.type_name(),
        "href": href,
        "children": children,
    })
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

        url
    }

    fn should_redirect(&self) -> bool {
        self.degree.is_none() || self.direction.is_none()
    }
}

#[instrument(
    level = "debug",
    skip(app, params),
    fields(genotype_id = ?params.genotype_id, degree = ?params.degree, direction = ?params.direction)
)]
async fn lineage_get(
    State(app): State<Arc<App>>,
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

    let (lineage_records, lineage_evaluations) = match genotype_id.as_ref() {
        Some(genotype_id) => {
            let result = match direction {
                Direction::Ancestors => {
                    app.services()
                        .genotype_explorer()
                        .get_ancestors(genotype_id, degree)
                        .await
                }
                Direction::Descendants => {
                    app.services()
                        .genotype_explorer()
                        .get_descendants(genotype_id, degree)
                        .await
                }
            };

            match result {
                Ok(records) => {
                    let ids: Vec<Uuid> = records.iter().map(|g| g.id()).collect();
                    let evals = app
                        .repositories()
                        .evaluations()
                        .search_evaluations(
                            &crate::services::evaluation::SearchEvaluationsFilter::default()
                                .with_genotype_ids(ids),
                        )
                        .await
                        .unwrap_or_default();
                    let evaluations: HashMap<Uuid, f64> = evals
                        .into_iter()
                        .map(|e| (*e.genotype_id(), e.fitness()))
                        .collect();
                    (records, evaluations)
                }
                Err(err) => {
                    tracing::error!(error = %err, "failed to fetch lineage records");
                    (Vec::new(), HashMap::new())
                }
            }
        }
        None => (Vec::new(), HashMap::new()),
    };

    let lineage_json = build_lineage_json(
        genotype_id.as_ref(),
        &lineage_records,
        &lineage_evaluations,
        degree,
        direction,
    );

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
