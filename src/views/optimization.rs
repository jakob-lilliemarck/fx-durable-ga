use crate::services::optimization::Request;
use axum::http::Uri;
use uuid::Uuid;

#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct Optimization {
    id: String,
    type_name: String,
}

impl From<&Request> for Optimization {
    fn from(value: &Request) -> Self {
        Self {
            id: value.id.to_string(),
            type_name: value.type_name.to_owned(),
        }
    }
}

pub struct OptimizationListView {
    optimizations: Vec<Request>,
    optimization_type_names: Vec<String>,
    cursor: Option<Uuid>,
    search: String,
    is_partial: bool,
}

impl OptimizationListView {
    pub fn new() -> Self {
        Self {
            optimizations: Vec::new(),
            optimization_type_names: Vec::new(),
            cursor: None,
            search: String::new(),
            is_partial: false,
        }
    }

    pub fn with_optimization(&mut self, optimizations: Vec<Request>) {
        self.optimizations = optimizations;
    }

    pub fn with_optimization_type_names(&mut self, type_names: Vec<String>) {
        self.optimization_type_names = type_names;
    }

    pub fn with_cursor(&mut self, cursor: Uuid) {
        self.cursor = Some(cursor);
    }

    pub fn with_search(&mut self, search: String) {
        self.search = search;
    }

    pub fn with_is_partial(&mut self) {
        self.is_partial = true;
    }

    fn items(&self) -> maud::Markup {
        maud::html! {
            @for (i, optimization) in self.optimizations.iter().enumerate() {
                @let is_last = i == self.optimizations.len() - 1;
                li
                    class="fx-flex"
                    hx-get=[is_last.then(|| self.cursor.map(|c| format!("?cursor={}", c))).flatten()]
                    hx-trigger=[is_last.then_some("intersect once")]
                    hx-target=[is_last.then_some("#optimization-list-items")]
                    hx-swap=[is_last.then_some("beforeend")]
                {
                    a class="card" href=(format!("/optimizations/{id}", id = optimization.id.to_string())) {
                        div class="card-title" {
                            span { (format!("{}:", optimization.type_name)) }
                            span { (optimization.id) }
                        }
                        time
                            class="card-timestamp"
                            datetime=(optimization.requested_at.to_rfc3339()) {
                                (optimization.requested_at.format("%b %d %Y, %H:%M"))
                        }
                    }
                }
            }
        }
    }
}

impl maud::Render for OptimizationListView {
    fn render(&self) -> maud::Markup {
        maud::html! {
            @if self.is_partial {
                (self.items())
            } @else {
                div id="optimization-list" class="layout" {
                    div class="layout-main" {
                        div class="heading" { "Optimizations" }
                        div class="fx-row" {
                            form
                            class="fx-form"
                            hx-get="/optimizations"
                            hx-target="#optimization-list-items"
                            hx-swap="innerHTML"
                            hx-trigger="change delay:200ms from:input[type=date], keyup delay:200ms from:input[name=search]" {
                                label for="optimizations_search_input" class="fx-flex-grow" hidden { "Search" }
                                input
                                    id="optimizations_search_input"
                                    name="search"
                                    placeholder="Search"
                                    value=(self.search)
                                    class="fx-flex-grow" {}
                                div class="fx-row-group" {
                                    label for="optimizations_since_input" hidden { "Since" }
                                    input id="optimizations_since_input" class="fx-row-item" name="since" type="date" {}
                                    label for="optimizations_until_input" hidden { "Until" }
                                    input id="optimizations_until_input" class="fx-row-item" name="until" type="date" {}
                                }
                            }

                            button
                                onclick="document.getElementById('new-optimization-dialog').showModal()"
                                class="fx-button primary fx-row-item" {
                                "New"
                            }

                            dialog id="new-optimization-dialog" onclose="this.querySelector('form').reset()" {
                                div class="heading" { "New optimization" }

                                form class="fx-flex fx-form" action="/optimizations" method="post" autocomplete="off" {
                                    label for="type_name_input" { "Type" }
                                    select id="type_name_input" name="type_name" {
                                        @for type_name in &self.optimization_type_names {
                                            option value=(type_name) { (type_name) }
                                        }
                                    }
                                    label for="max_evaluations_input" { "Max evaluations" }
                                    input id="max_evaluations_input" type="number" name="max_evaluations" min="1" value="2000" required;
                                    label for="population_size_input" { "Population size" }
                                    input id="population_size_input" type="number" name="population_size" min="1" value="30" required;
                                    label for="selection_interval_input" { "Selection interval" }
                                    input id="selection_interval_input" type="number" name="selection_interval" min="1" value="30";
                                    label for="tournament_size_input" { "Tournament size" }
                                    input id="tournament_size_input" type="number" name="tournament_size" min="1";
                                    label for="goal_direction_input" { "Goal direction" }
                                    select id="goal_direction" name="goal_direction" {
                                        option value="minimize" { "Minimize" }
                                        option value="maximize" { "Maximize" }
                                    }
                                    label for="goal_threshold_input" { "Goal threshold" }
                                    input id="goal_threshold_input" type="number" name="goal_threshold" step="any" required;
                                    button class="fx-button" formmethod="dialog" formnovalidate { "Cancel" }
                                    button class="fx-button primary" type="submit" { "Create" }
                                }
                            }
                        }

                        div class="fx-flex block" {
                            @if self.optimizations.is_empty() {
                                "No optimizations found"
                            } @else {
                                ul id="optimization-list-items" class="fx-flex optimizations" {
                                    (self.items())
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl serde::Serialize for OptimizationListView {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let optimizations: Vec<Optimization> =
            self.optimizations.iter().map(Optimization::from).collect();
        optimizations.serialize(serializer)
    }
}

impl schemars::JsonSchema for OptimizationListView {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        Vec::<Optimization>::schema_name()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<Optimization>::json_schema(generator)
    }
}

pub struct OptimizationView {
    uri: Uri,
    optimization: Request,
}

impl OptimizationView {
    const NS_SEPARATOR: &str = "--";

    pub fn new(uri: Uri, optimization: Request) -> Self {
        Self { uri, optimization }
    }

    fn base_url(&self) -> String {
        self.uri.path().to_string()
    }

    fn namespaced_params(&self, namespace: &str) -> String {
        self.uri
            .query()
            .unwrap_or_default()
            .split('&')
            .filter_map(|pair| {
                let (k, v) = pair.split_once('=')?;
                let stripped = k.strip_prefix(&format!("{}{}", namespace, Self::NS_SEPARATOR))?;
                Some(format!("{stripped}={v}"))
            })
            .collect::<Vec<_>>()
            .join("&")
    }
}

impl maud::Render for OptimizationView {
    fn render(&self) -> maud::Markup {
        let knn_query_params = self.namespaced_params("knn");
        let knn_url = format!("{}/knn?{}", self.base_url(), knn_query_params);

        let fitness_url = format!("{}/fitness", self.base_url());

        let population_url = format!("{}/population", self.base_url());

        let genotypes_url = format!("{}/genotypes", self.base_url());

        maud::html! {
            div id="optimization" class="layout" {
                div class="layout-header" {
                    h1 { (self.optimization.id.to_string()) }
                    p class="subtitle" { (self.optimization.type_name) }
                }

                div class="layout-main" {
                    div id="fitness" class="fx-flex block"
                        hx-get=(fitness_url)
                        hx-trigger="load" {}

                    div id="diversity" class="fx-flex block"
                        hx-get=(knn_url)
                        hx-trigger="load" {}
                }

                div class="fx-flex layout-aside" {
                    div class="fx-flex block" {
                        div class="heading" { "Progress" }
                        div id="population"
                            class="fx-flex"
                            hx-get=(population_url)
                            hx-trigger="load" {}
                    }

                    div id="optimization-genotypes" class="fx-flex genotypes block"
                        hx-get=(genotypes_url)
                        hx-trigger="load" {}
                }
            }
        }
    }
}

impl serde::Serialize for OptimizationView {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Optimization::from(&self.optimization).serialize(serializer)
    }
}
