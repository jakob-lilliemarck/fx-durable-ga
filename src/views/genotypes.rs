use crate::repositories::genotypes::Genotype;
use serde::Serialize;
use std::collections::HashMap;
use uuid::Uuid;

pub struct GenotypeListView {
    url: String,
    genotypes: Vec<Genotype>,
    fitness_map: HashMap<Uuid, f64>,
    max_fitness: Option<f64>,
    min_fitness: Option<f64>,
    cursor: Option<Uuid>,
    is_partial: bool,
}

impl GenotypeListView {
    pub fn new(url: String) -> Self {
        Self {
            url,
            genotypes: Vec::new(),
            fitness_map: HashMap::new(),
            min_fitness: None,
            max_fitness: None,
            cursor: None,
            is_partial: false,
        }
    }

    pub fn with_genotypes(&mut self, genotypes: Vec<Genotype>) {
        self.genotypes = genotypes;
    }

    pub fn with_fitness_map(&mut self, fitness_map: HashMap<Uuid, f64>) {
        self.fitness_map = fitness_map;
    }

    pub fn with_max_fitness(&mut self, max_fitness: f64) {
        self.max_fitness = Some(max_fitness);
    }

    pub fn with_min_fitness(&mut self, min_fitness: f64) {
        self.min_fitness = Some(min_fitness);
    }

    pub fn with_cursor(&mut self, cursor: Uuid) {
        self.cursor = Some(cursor);
    }

    pub fn with_is_partial(&mut self) {
        self.is_partial = true
    }

    fn items(&self) -> maud::Markup {
        maud::html! {
            @for (i, genotype) in self.genotypes.iter().enumerate() {
                @let fitness = self.fitness_map.get(&genotype.id()).copied();
                @let is_last = i == self.genotypes.len() - 1;
                @let has_next_page = is_last && self.cursor.is_some();
                li
                    class="fx-flex"
                    hx-get=[has_next_page.then(|| self.cursor.map(|c| format!("{url}?cursor={c}", url = self.url))).flatten()]
                    hx-include=[has_next_page.then_some("#search_genotypes_form")]
                    hx-trigger=[has_next_page.then_some("intersect once")]
                    hx-target=[has_next_page.then_some("#genotypes-list-items")]
                    hx-swap=[has_next_page.then_some("beforeend")]
                {
                    a class="card" href=(format!("/lineage?genotype_id={}&direction=descendants&degree=3", genotype.id)) {
                        div class="card-title" {
                            span {"genotype:" }
                            span {(genotype.id) }
                        }
                        time
                            class="card-timestamp"
                            datetime=(genotype.generated_at.to_rfc3339()) {
                                (genotype.generated_at.format("%b %d %Y, %H:%M"))
                        }
                        div title="fitness" {
                            @match fitness {
                                Some(fitness) => (format!("{:.3}", fitness)),
                                None => ("Not evaluated"),
                            }
                        }
                    }
                }
            }
        }
    }
}

impl maud::Render for GenotypeListView {
    fn render(&self) -> maud::Markup {
        maud::html! {
            @if self.is_partial {
                (self.items())
            } @else {
                section id="genotypes_list" class="fx-flex" {
                    div class="heading" { "Genotypes" }
                    div class="fx-row" {
                        form
                            id="search_genotypes_form"
                            hx-get=(self.url)
                            hx-target="#genotypes-list-items"
                            hx-swap="innerHTML"
                            hx-trigger="input delay:200ms"
                            class="fx-form fx-flex-grow"
                        {
                            label for="search_genotypes_search_input" hidden { "Search" }
                            input id="search_genotypes_search_input" name="search" class="fx-flex-grow" {}
                            div class="fx-row-group" {
                                label for="search_genotypes_min_fitness_input" hidden { "Minimum fitness" }
                                input
                                    id="search_genotypes_min_fitness_input"
                                    title="Minimum fitness"
                                    name="min_fitness"
                                    type="number"
                                    step="any"
                                    lang="en" {}
                                label for="search_genotypes_max_fitness_input" hidden { "Maximum fitness" }
                                input
                                    id="search_genotypes_max_fitness_input"
                                    title="Maximum fitness"
                                    name="max_fitness"
                                    type="number"
                                    step="any"
                                    lang="en" {}
                            }
                        }
                    }

                    @if self.genotypes.is_empty() {
                       div { "No genotypes found" }
                    } @else {
                        ul id="genotypes-list-items" class="fx-flex" {
                            (self.items())
                        }
                    }
                }
            }
        }
    }
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct GenotypeWithFitness {
    id: Uuid,
    genome: serde_json::Value,
    fitness: Option<f64>,
}

impl serde::Serialize for GenotypeListView {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let genotypes: Vec<GenotypeWithFitness> = self
            .genotypes
            .iter()
            .map(|genotype| GenotypeWithFitness {
                id: genotype.id(),
                genome: genotype.genome(),
                fitness: self.fitness_map.get(&genotype.id()).copied(),
            })
            .collect();

        genotypes.serialize(serializer)
    }
}
