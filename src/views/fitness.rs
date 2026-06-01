use crate::services::evaluation::AggregatedFitness;
use serde::Serialize;

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct FitnessBin {
    bin_index: i64,
    min_fitness: f64,
    avg_fitness: f64,
}

pub struct FitnessView {
    fitness_bins: Vec<FitnessBin>,
}

impl FitnessView {
    pub fn new() -> Self {
        Self {
            fitness_bins: Vec::new(),
        }
    }

    pub fn with_fitness_bins(&mut self, aggregated_fitness: Vec<AggregatedFitness>) {
        self.fitness_bins = aggregated_fitness
            .into_iter()
            .map(|row| FitnessBin {
                bin_index: row.bin_index,
                min_fitness: row.min_fitness,
                avg_fitness: row.avg_fitness,
            })
            .collect();
    }

    fn chart_config(&self) -> Option<String> {
        if self.fitness_bins.is_empty() {
            return None;
        }

        let labels: Vec<i64> = self.fitness_bins.iter().map(|bin| bin.bin_index).collect();
        let min_values: Vec<f64> = self
            .fitness_bins
            .iter()
            .map(|bin| bin.min_fitness)
            .collect();
        let avg_values: Vec<f64> = self
            .fitness_bins
            .iter()
            .map(|bin| bin.avg_fitness)
            .collect();

        let config = serde_json::json!({
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Min Fitness",
                        "data": min_values,
                        "borderColor": "#1f77b4",
                        "borderWidth": 2,
                        "fill": false,
                        "tension": 0,
                        "yAxisID": "yMin"
                    },
                    {
                        "label": "Avg Fitness",
                        "data": avg_values,
                        "borderColor": "#ff7f0e",
                        "borderWidth": 2,
                        "fill": false,
                        "tension": 0,
                        "yAxisID": "yAvg"
                    }
                ]
            },
            "options": {
                "responsive": true,
                "maintainAspectRatio": false,
                "animation": false,
                "interaction": {
                    "intersect": false,
                    "mode": "index"
                },
                "plugins": {
                    "legend": { "display": true }
                },
                "scales": {
                    "x": {
                        "title": { "display": true, "text": "Bin" }
                    },
                    "yMin": {
                        "type": "linear",
                        "position": "left",
                        "title": { "display": true, "text": "Min Fitness" }
                    },
                    "yAvg": {
                        "type": "linear",
                        "position": "right",
                        "title": { "display": true, "text": "Avg Fitness" },
                        "grid": { "drawOnChartArea": false }
                    }
                }
            }
        });

        serde_json::to_string(&config).ok()
    }
}

impl maud::Render for FitnessView {
    fn render(&self) -> maud::Markup {
        maud::html! {
            div class="heading" { "Fitness" }
            @match self.chart_config() {
                None => {
                    div class="fx-flex" { "no data" }
                }
                Some(config) => {
                    section class="chart fx-flex" {
                        canvas
                        data-line-chart
                        data-line-config=(config)
                        {}
                    }
                }
            }
        }
    }
}

impl serde::Serialize for FitnessView {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.fitness_bins.serialize(serializer)
    }
}
