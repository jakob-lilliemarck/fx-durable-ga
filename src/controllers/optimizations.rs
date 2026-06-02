use crate::repositories::genotypes as data;
use crate::services::evaluation::{GetEvaluationStatsFilter, SearchEvaluationsFilter};
use crate::services::indexing::Digest;
use crate::{
    bootstrap::App,
    controllers::models::{form_or_json::FormOrJson, namespaced_query::NamespacedQuery},
    repositories::genotypes::SearchGenotypesFilter,
    services::evaluation::GetAggregatedFitnessFilter,
    services::optimization::{self, FitnessGoal, Schedule, SearchRequestsFilter, Selector},
    views::{
        self, diversity::KnnView, errors::ErrorResponse, fitness::FitnessView,
        population::PopulationView,
    },
};
use aide::axum::ApiRouter;
use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, Uri},
    response::{Html, IntoResponse, Redirect, Response},
};
use chrono::{NaiveDate, NaiveTime};
use maud::PreEscaped;
use maud::{Markup, html};
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

const PAGE_LENGTH: i64 = 20;

pub fn router(app: Arc<App>) -> ApiRouter {
    ApiRouter::new()
        .api_route(
            "/optimizations",
            aide::axum::routing::post_with(post_optimization, |op| {
                op.tag("optimizations")
                    .summary("Crate optimization")
                    .response::<200, Json<Uuid>>()
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .api_route(
            "/optimizations",
            aide::axum::routing::get_with(get_optimization_list, |op| {
                op.tag("optimizations")
                    .summary("List optimizations")
                    .response::<200, Json<Vec<views::optimization::Optimization>>>()
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .api_route(
            "/optimizations/{id}",
            aide::axum::routing::get_with(get_optimization, |op| {
                op.tag("optimizations")
                    .summary("Single optimization")
                    .response::<200, Json<views::optimization::Optimization>>()
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .api_route(
            "/optimizations/{id}/population",
            aide::axum::routing::get_with(get_population, |op| {
                op.tag("optimizations")
                    .summary("Optimization population data")
                    .response::<200, Json<views::population::Population>>()
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .api_route(
            "/optimizations/{id}/fitness",
            aide::axum::routing::get_with(get_fitness, |op| {
                op.tag("optimizations")
                    .summary("Aggregated fitness")
                    .response::<200, Json<Vec<views::fitness::FitnessBin>>>()
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .api_route(
            "/optimizations/{id}/knn",
            aide::axum::routing::get_with(get_diversity, |op| {
                op.tag("optimizations")
                    .summary("Aggregated k-nearest neighbor")
                    .response::<200, Json<Vec<views::diversity::KnnBin>>>()
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .api_route(
            "/optimizations/{id}/genotypes",
            aide::axum::routing::get_with(get_genotypes, |op| {
                op.tag("optimizations")
                    .summary("Optimization genotypes")
                    .response::<200, Json<Vec<views::genotypes::GenotypeWithFitness>>>()
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .api_route(
            "/optimizations/index",
            aide::axum::routing::post_with(trigger_indexing, |op| {
                op.tag("optimizations")
                    .summary("Trigger indexing")
                    .response::<201, ()>()
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .with_state(app)
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GoalDirection {
    Minimize,
    Maximize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PostOptimizationPayload {
    pub type_name: String,
    pub max_evaluations: u32,
    pub population_size: u32,
    pub selection_interval: Option<u32>,
    pub tournament_size: Option<usize>,
    pub goal_direction: GoalDirection,
    pub goal_threshold: f64,
}

#[axum::debug_handler]
async fn post_optimization(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    input: FormOrJson<PostOptimizationPayload>,
) -> Response {
    let input = input.into_inner();

    let goal_result = match input.goal_direction {
        GoalDirection::Maximize => FitnessGoal::maximize(input.goal_threshold),
        GoalDirection::Minimize => FitnessGoal::minimize(input.goal_threshold),
    };

    let goal = match goal_result {
        Ok(g) => g,
        Err(err) => return RenderService.respond_error(&headers, StatusCode::BAD_REQUEST, err),
    };

    let schedule = Schedule::new(
        input.max_evaluations,
        input.population_size,
        input.selection_interval.unwrap_or(input.population_size),
    );

    let selector = match input.tournament_size {
        Some(tournament_size) => Selector::tournament(tournament_size),
        None => Selector::roulette(),
    };

    let optimization_id = match app
        .services()
        .optimization()
        .request_new(input.type_name, goal, schedule, selector)
        .await
    {
        Ok(optimization_id) => optimization_id,
        Err(err @ optimization::Error::UnknownTypeError { .. }) => {
            return RenderService.respond_error(&headers, StatusCode::BAD_REQUEST, err);
        }
        Err(err) => {
            tracing::error!(err=%err, "Failed to create optimization");
            return RenderService.respond_error(&headers, StatusCode::INTERNAL_SERVER_ERROR, err);
        }
    };

    if RenderService::wants_json(&headers) {
        return (StatusCode::CREATED, Json(optimization_id)).into_response();
    }

    if RenderService::wants_htmx(&headers) {
        return (
            StatusCode::OK,
            [("HX-Redirect", format!("/optimizations/{}", optimization_id))],
        )
            .into_response();
    }

    Redirect::to(&format!("/optimizations/{}", optimization_id)).into_response()
}

/// Deserializes optional date params from string, treating empty string "" like None instead of error
fn deserialize_optional_date<'de, D>(deserializer: D) -> Result<Option<NaiveDate>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: Option<String> = Option::deserialize(deserializer)?;
    match s {
        None => Ok(None),
        Some(v) if v.is_empty() => Ok(None),
        Some(v) => NaiveDate::parse_from_str(&v, "%Y-%m-%d")
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct GetOptimizationListQuery {
    cursor: Option<Uuid>,
    search: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_date")]
    since: Option<NaiveDate>,
    #[serde(default, deserialize_with = "deserialize_optional_date")]
    until: Option<NaiveDate>,
}

async fn get_optimization_list(
    State(app): State<Arc<App>>,
    NamespacedQuery(params): NamespacedQuery<GetOptimizationListQuery>,
    headers: HeaderMap,
) -> Response {
    let mut view = views::optimization::OptimizationListView::new();
    let mut filter = SearchRequestsFilter::default();

    if let Some(search) = params.search {
        filter = filter.with_search(search.to_string());
        view.with_search(search.to_string())
    }

    if let Some(since) = params.since {
        filter = filter.with_since(since.and_time(NaiveTime::MIN).and_utc());
    }

    if let Some(until) = params.until {
        filter = filter.with_until(until.and_time(NaiveTime::MIN).and_utc());
    }

    if let Some(cursor) = params.cursor {
        filter = filter.with_cursor(&cursor);
    }

    let optimizations = match app
        .repositories()
        .requests()
        .search_requests(&filter, PAGE_LENGTH)
        .await
    {
        Ok(o) => o,
        Err(err) => {
            tracing::error!(error = %err, "failed to fetch optimization list");
            return RenderService.respond_error(&headers, StatusCode::INTERNAL_SERVER_ERROR, err);
        }
    };

    let optimization_type_names = app
        .services()
        .optimization()
        .get_registered_type_names()
        .await;

    if headers.get("HX-Request").is_some() {
        view.with_is_partial();
    }

    if let Some(last) = optimizations.get(PAGE_LENGTH as usize - 1) {
        view.with_cursor(last.id);
    }

    view.with_optimization(optimizations);

    view.with_optimization_type_names(optimization_type_names);

    RenderService.respond(&headers, StatusCode::OK, &view)
}

async fn get_optimization(
    State(app): State<Arc<App>>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    uri: Uri,
) -> Response {
    let optimization = match app.repositories().requests().get_request(id).await {
        Ok(o) => o,
        Err(err) => {
            tracing::error!(error = %err, "failed to fetch optimization");
            return RenderService.respond_error(&headers, StatusCode::INTERNAL_SERVER_ERROR, err);
        }
    };

    let view = views::optimization::OptimizationView::new(uri, optimization);

    RenderService.respond(&headers, StatusCode::OK, &view)
}

async fn get_population(
    State(app): State<Arc<App>>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
) -> Response {
    let request = match app.repositories().requests().get_request(id).await {
        Ok(r) => r,
        Err(err) => {
            tracing::error!(error = %err, "failed to fetch request");
            return RenderService.respond_error(&headers, StatusCode::INTERNAL_SERVER_ERROR, err);
        }
    };

    let genotype_pop = match app.repositories().genotypes().get_population(&id).await {
        Ok(p) => p,
        Err(err) => {
            error!(error = %err, "failed to fetch population");
            return RenderService.respond_error(&headers, StatusCode::INTERNAL_SERVER_ERROR, err);
        }
    };

    let eval_pop = app
        .repositories()
        .evaluations()
        .get_evaluation_stats(
            &GetEvaluationStatsFilter::default().with_request_id(id),
            i64::MAX,
        )
        .await
        .ok();

    let population = data::Population::new(
        id,
        genotype_pop.total_genotypes(),
        eval_pop
            .as_ref()
            .map(|p| p.evaluated_genotypes())
            .unwrap_or(0),
        genotype_pop.current_generation(),
        eval_pop.as_ref().and_then(|p| p.min_fitness()),
        eval_pop.as_ref().and_then(|p| p.max_fitness()),
    );

    let view = PopulationView::new(population, request);

    RenderService.respond(&headers, StatusCode::OK, &view)
}

pub async fn get_fitness(
    State(app): State<Arc<App>>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
) -> Response {
    let request = match app.repositories().requests().get_request(id).await {
        Ok(request) => request,
        Err(err) => {
            error!(error = %err, "failed to fetch request");
            return RenderService.respond_error(
                &headers,
                StatusCode::INTERNAL_SERVER_ERROR,
                err.to_string(),
            );
        }
    };

    let eval_pop = app
        .repositories()
        .evaluations()
        .get_evaluation_stats(
            &GetEvaluationStatsFilter::default().with_request_id(id),
            i64::MAX,
        )
        .await
        .ok();

    let bin_size = i64::from(request.schedule.selection_interval);
    let evaluated = eval_pop
        .as_ref()
        .map(|p| p.evaluated_genotypes())
        .unwrap_or(0)
        .max(0);
    let limit = (evaluated + bin_size - 1) / bin_size;

    let aggregated_fitness = match app
        .repositories()
        .evaluations()
        .get_aggregated_fitness(
            &id,
            bin_size,
            limit.max(1),
            &GetAggregatedFitnessFilter::default(),
        )
        .await
    {
        Ok(aggregated) => aggregated,
        Err(err) => {
            error!(error = %err, "failed to fetch aggregated fitness");
            return RenderService.respond_error(
                &headers,
                StatusCode::INTERNAL_SERVER_ERROR,
                err.to_string(),
            );
        }
    };

    let mut view = FitnessView::new();
    view.with_fitness_bins(aggregated_fitness);

    RenderService.respond(&headers, StatusCode::OK, &view)
}

#[derive(Deserialize, schemars::JsonSchema)]
struct DiversityQuery {
    indexer_id: Option<String>,
    k: Option<i32>,
}

async fn get_diversity(
    State(app): State<Arc<App>>,
    Path(id): Path<Uuid>,
    NamespacedQuery(params): NamespacedQuery<DiversityQuery>,
    headers: HeaderMap,
) -> Response {
    let url = format!("/optimizations/{id}/knn", id = id);
    let mut view = KnnView::new(url);

    let request = match app.repositories().requests().get_request(id).await {
        Ok(request) => request,
        Err(err) => {
            error!(error = %err, "failed to fetch request");
            return RenderService.respond_error(
                &headers,
                StatusCode::INTERNAL_SERVER_ERROR,
                err.to_string(),
            );
        }
    };

    let indexer_ids: Vec<String> = match app
        .services()
        .genotype_indexing()
        .get_indexers_of_request(&request)
        .await
    {
        Ok(digests) => digests.into_iter().map(|d| d.to_string()).collect(),
        Err(err) => {
            error!(error = %err, "failed to fetch indexers for request");
            return RenderService.respond_error(
                &headers,
                StatusCode::INTERNAL_SERVER_ERROR,
                err.to_string(),
            );
        }
    };

    let indexer_id_str = params
        .indexer_id
        .filter(|indexer_id_str| indexer_ids.contains(&indexer_id_str));

    view = view.with_indexer_options(indexer_ids);

    let Some(indexer_id_str) = indexer_id_str else {
        return RenderService.respond(&headers, StatusCode::OK, &view);
    };

    let indexer_id = match Digest::from_hex(&indexer_id_str) {
        Ok(indexer_id) => indexer_id,
        Err(err) => {
            return RenderService.respond_error(
                &headers,
                StatusCode::INTERNAL_SERVER_ERROR,
                err.to_string(),
            );
        }
    };

    let is_pending = match app
        .services()
        .genotype_indexing()
        .is_indexing_pending(&indexer_id, id)
        .await
    {
        Ok(indexer_id) => indexer_id,
        Err(err) => {
            return RenderService.respond_error(
                &headers,
                StatusCode::INTERNAL_SERVER_ERROR,
                err.to_string(),
            );
        }
    };

    view = view.with_selected_indexer(indexer_id_str, is_pending);

    let generation_count = request.schedule.max_evaluations / request.schedule.selection_interval;
    let tag_groups: Vec<Vec<String>> = (1..=generation_count)
        .map(|g| {
            vec![
                format!("request_id:{}", id),
                format!("indexer_id:{}", indexer_id),
                format!("generation_id:{}", g),
            ]
        })
        .collect();

    let tag_groups_refs: Vec<Vec<&str>> = tag_groups
        .iter()
        .map(|group| group.iter().map(|s| s.as_str()).collect())
        .collect();

    let tag_groups_slices: Vec<&[&str]> = tag_groups_refs
        .iter()
        .map(|group| group.as_slice())
        .collect();

    let binned_ids = match app
        .repositories()
        .embeddings()
        .get_embeddings_by_tag_groups(&tag_groups_slices)
        .await
    {
        Ok(ids) => ids,
        Err(err) => {
            error!(error = %err, "failed to fetch embeddings by tag groups");
            return RenderService.respond_error(
                &headers,
                StatusCode::INTERNAL_SERVER_ERROR,
                err.to_string(),
            );
        }
    };

    let k = params.k.unwrap_or(1);

    let diversity_bins = match app
        .repositories()
        .embeddings()
        .get_knn_diversity_stats(&binned_ids, k)
        .await
    {
        Ok(bins) => bins,
        Err(err) => {
            error!(error = %err, "failed to fetch knn diversity stats");
            return RenderService.respond_error(
                &headers,
                StatusCode::INTERNAL_SERVER_ERROR,
                err.to_string(),
            );
        }
    };

    view = view.with_data(diversity_bins).with_k(k);

    RenderService.respond(&headers, StatusCode::OK, &view)
}

#[derive(Deserialize, schemars::JsonSchema)]
struct GenotypesQuery {
    cursor: Option<Uuid>,
    search: Option<String>,
    min_fitness: Option<f64>,
    max_fitness: Option<f64>,
}

async fn get_genotypes(
    State(app): State<Arc<App>>,
    Path(id): Path<Uuid>,
    NamespacedQuery(query): NamespacedQuery<GenotypesQuery>,
    headers: HeaderMap,
) -> Response {
    let genotypes_url = format!("/optimizations/{}/genotypes", id);

    let mut view = views::genotypes::GenotypeListView::new(genotypes_url);
    let mut filter = SearchGenotypesFilter::default();

    if let Some(search) = query.search {
        filter = filter.with_search(search)
    }

    if let Some(cursor) = query.cursor {
        filter = filter.with_cursor(cursor);
        view.with_cursor(cursor);
    }

    let genotypes = match app
        .repositories()
        .genotypes()
        .search_genotypes(&filter, PAGE_LENGTH)
        .await
    {
        Ok(genotypes) => genotypes,
        Err(err) => {
            error!(error = %err, "failed to fetch genotypes");
            return RenderService.respond_error(
                &headers,
                StatusCode::INTERNAL_SERVER_ERROR,
                err.to_string(),
            );
        }
    };

    let ids: Vec<Uuid> = genotypes.iter().map(|g| g.id()).collect();
    let evals = app
        .repositories()
        .evaluations()
        .search_evaluations(&SearchEvaluationsFilter::default().with_genotype_ids(ids))
        .await
        .unwrap_or_default();
    let fitness_map: HashMap<Uuid, f64> = evals
        .into_iter()
        .map(|e| (*e.genotype_id(), e.fitness()))
        .collect();

    if headers
        .get("HX-Target")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == "genotypes-list-items")
        .unwrap_or(false)
    {
        view.with_is_partial();
    }

    if let Some(genotype) = genotypes.get(PAGE_LENGTH as usize - 1) {
        view.with_cursor(genotype.id());
    }

    view.with_genotypes(genotypes);
    view.with_fitness_map(fitness_map);

    RenderService.respond(&headers, StatusCode::OK, &view)
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct TriggerIndexingForm {
    request_id: Uuid,
}

async fn trigger_indexing(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    input: FormOrJson<TriggerIndexingForm>,
) -> Response {
    let form = input.into_inner();
    let filter = SearchGenotypesFilter::default().with_request_id(form.request_id);

    if let Err(err) = app
        .services()
        .genotype_indexing()
        .backfill_missing_genotypes(filter)
        .await
    {
        error!(error = %err, "failed to trigger indexing");
        return RenderService.respond_error(&headers, StatusCode::INTERNAL_SERVER_ERROR, err);
    }

    if RenderService::wants_json(&headers) {
        return StatusCode::CREATED.into_response();
    }
    Redirect::to(&format!("/optimizations?id={}", form.request_id)).into_response()
}

/// Rendering infra
struct RenderService;

impl RenderService {
    const STYLES: &str = include_str!("../../public/styles.css");
    const LINE_CHART: &str = include_str!("../../public/charts/line.js");
    const QUERYPARAMS: &str = include_str!("../../public/charts/queryparams.js");

    fn respond<T>(&self, headers: &HeaderMap, status: StatusCode, view: &T) -> Response
    where
        T: maud::Render + serde::Serialize,
    {
        if Self::wants_json(headers) {
            return (status, Json(view)).into_response();
        }
        if Self::wants_htmx(headers) {
            return (status, Html(view.render().into_string())).into_response();
        }
        (
            status,
            Html(Self::full_page_markup(view.render()).into_string()),
        )
            .into_response()
    }

    fn respond_error(
        &self,
        headers: &HeaderMap,
        status: StatusCode,
        message: impl std::fmt::Display,
    ) -> Response {
        let message = message.to_string();

        if Self::wants_json(headers) {
            return (status, Json(ErrorResponse::new(message))).into_response();
        }

        if Self::wants_htmx(headers) {
            return (
                status,
                Html(Self::error_fragment_markup(&message).into_string()),
            )
                .into_response();
        }

        (
            status,
            Html(Self::full_page_markup(Self::error_page_markup()).into_string()),
        )
            .into_response()
    }

    fn wants_json(headers: &HeaderMap) -> bool {
        headers
            .get(axum::http::header::ACCEPT)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.contains("application/json"))
    }

    fn wants_htmx(headers: &HeaderMap) -> bool {
        headers.contains_key("HX-Request")
    }

    fn full_page_markup(content: Markup) -> Markup {
        let markup = html! {
            html {
                head {
                    meta name="viewport" content="width=device-width, initial-scale=1" {}
                    script src="https://unpkg.com/htmx.org@2.0.8" {}
                    script src="https://cdn.jsdelivr.net/npm/chart.js@4" {}
                    style { (PreEscaped(Self::STYLES)) }
                    script { (PreEscaped(Self::LINE_CHART)) }
                    script { (PreEscaped(Self::QUERYPARAMS)) }
                }
                body {
                    (content)
                }
            }
        };
        markup
    }

    fn error_fragment_markup(message: &str) -> Markup {
        html! {
            div role="status" { "Error: " (message) }
        }
    }

    fn error_page_markup() -> Markup {
        html! {
            main {
                h1 { "500 Internal Server Error" }
            }
        }
    }
}
