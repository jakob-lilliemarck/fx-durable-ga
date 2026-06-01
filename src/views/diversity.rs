use crate::services::indexing::embeddings;
use maud::Markup;
use serde::Serialize;

/// Provides a widget to select indexer, display knn statistics charts and trigger indexing
pub struct KnnView {
    url: String,
    selected_indexer: Option<(String, bool)>,
    k: Option<i32>,
    indexer_options: Vec<String>,
    knn_bins: Vec<embeddings::AggregatedDiversity>,
}

impl KnnView {
    pub fn new(url: String) -> Self {
        Self {
            url,
            k: None,
            selected_indexer: None,
            indexer_options: Vec::new(),
            knn_bins: Vec::new(),
        }
    }

    pub fn with_selected_indexer(mut self, indexer_id: String, indexing_pending: bool) -> Self {
        self.selected_indexer = Some((indexer_id, indexing_pending));
        self
    }

    pub fn with_indexer_options(mut self, indexer_options: Vec<String>) -> Self {
        self.indexer_options = indexer_options;
        self
    }

    pub fn with_data(mut self, data: Vec<embeddings::AggregatedDiversity>) -> Self {
        self.knn_bins = data;
        self
    }

    pub fn with_k(mut self, k: i32) -> Self {
        self.k = Some(k);
        self
    }

    fn chart_config(&self) -> Option<String> {
        if self.knn_bins.is_empty() {
            tracing::warn!("this should never happen");
            return None;
        }

        let labels: Vec<i64> = self.knn_bins.iter().map(|bin| bin.bin_index + 1).collect();

        let avg_values: Vec<Option<f64>> = self
            .knn_bins
            .iter()
            .map(|bin| bin.avg_knn_distance)
            .collect();

        let min_values: Vec<Option<f64>> = self
            .knn_bins
            .iter()
            .map(|bin| bin.min_knn_distance)
            .collect();

        let max_values: Vec<Option<f64>> = self
            .knn_bins
            .iter()
            .map(|bin| bin.max_knn_distance)
            .collect();

        let config = serde_json::json!({
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Max K-NN Distance",
                        "data": max_values,
                        "borderColor": "rgba(31, 119, 180, 0.4)",
                        "borderWidth": 1,
                        "pointRadius": 0,
                        "fill": "+1",
                        "backgroundColor": "rgba(31, 119, 180, 0.1)",
                        "tension": 0
                    },
                    {
                        "label": "Avg K-NN Distance",
                        "data": avg_values,
                        "borderColor": "#1f77b4",
                        "borderWidth": 2,
                        "pointRadius": 2,
                        "fill": false,
                        "tension": 0
                    },
                    {
                        "label": "Min K-NN Distance",
                        "data": min_values,
                        "borderColor": "rgba(31, 119, 180, 0.4)",
                        "borderWidth": 1,
                        "pointRadius": 0,
                        "fill": false,
                        "tension": 0
                    }
                ]
            },
            "options": {
                "responsive": true,
                "maintainAspectRatio": false,
                "animation": false,
                "interaction": { "intersect": false, "mode": "index" },
                "plugins": { "legend": { "display": true } },
                "scales": {
                    "x": {
                        "title": { "display": true, "text": "Generation" }
                    },
                    "y": {
                        "type": "linear",
                        "position": "left",
                        "title": { "display": true, "text": "K-NN Distance (k=5)" }
                    }
                }
            }
        });

        serde_json::to_string(&config).ok()
    }
}

impl maud::Render for KnnView {
    fn render(&self) -> Markup {
        maud::html! {
            div id="knn" class="fx-flex" {
                div class="heading" { "K-nearest neighbor" }
                form class="fx-flex fx-form fx-form-inline" hx-get=(self.url)
                    hx-target="#knn"
                    hx-swap="outerHTML"
                    hx-trigger="change from:select, change delay:200ms from:input" {
                    select name="indexer_id" {
                        option value="" disabled selected[self.selected_indexer.is_none()] {
                            "Select an indexer"
                        }
                        @for indexer_id in &self.indexer_options {
                            option
                            value=(indexer_id)
                            selected[self.selected_indexer.as_ref().map(|(id,_)| id == indexer_id).unwrap_or(false)]
                            {
                                (indexer_id)
                            }
                        }
                    }
                    input type="number" name="k"
                        min="1"
                        value=(self.k.unwrap_or(1)) {}
                }

                @if let Some((_, pending)) = self.selected_indexer {
                    @if !pending && self.knn_bins.is_empty() {
                        "trigger indexing"
                    } @else if pending && self.knn_bins.is_empty() {
                        "indexing pending"
                    } @else {
                        @if let Some(config) = self.chart_config() {
                            section class="fx-flex chart" {
                                canvas data-line-chart data-line-config=(config) {}
                            }
                        } @else {
                            div class="fx-flex error" {
                                "Could not initialize chart"
                            }
                        }
                    }
                } @else {
                    "No indexer selected"
                }
            }
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct KnnBin {
    pub bin_index: i64,
    pub embedding_count: i64,
    pub min_knn_distance: Option<f64>,
    pub max_knn_distance: Option<f64>,
    pub avg_knn_distance: Option<f64>,
    pub sample_variance: Option<f64>,
    pub sample_std: Option<f64>,
}

impl From<embeddings::AggregatedDiversity> for KnnBin {
    fn from(a: embeddings::AggregatedDiversity) -> Self {
        Self {
            bin_index: a.bin_index,
            embedding_count: a.embedding_count,
            min_knn_distance: a.min_knn_distance,
            max_knn_distance: a.max_knn_distance,
            avg_knn_distance: a.avg_knn_distance,
            sample_variance: a.sample_variance,
            sample_std: a.sample_std,
        }
    }
}

impl serde::Serialize for KnnView {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.knn_bins.serialize(serializer)
    }
}
