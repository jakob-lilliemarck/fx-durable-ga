---
name: create-controller
description: |
  Use when creating a new controller module with HTTP handlers for an existing
  service. Covers the controller file, route definitions, handler signatures,
  content negotiation, view types, and registration. Do NOT use for services,
  repositories, or infrastructure code.
---

# Create Controller

## Prerequisites

The service this controller exposes must already exist. If creating a new
service, run the `scaffold-service` tool first (and `scaffold-repository` if
the service owns data).

## Step 1 — Create the controller file

Create `src/controllers/{resource}.rs` with:

```rust
use crate::bootstrap::App;
use crate::controllers::models::{FormOrJson, NamespacedQuery};
use crate::views::errors::ErrorResponse;
use aide::axum::{routing::get_with, ApiRouter};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;
use uuid::Uuid;

struct RenderService;

impl RenderService {
    fn respond<T>(&self, headers: &HeaderMap, status: StatusCode, view: &T) -> Response
    where
        T: maud::Render + serde::Serialize,
    {
        // JSON → Json(view), HX-Request → Html(view.render()), else → full HTML
        todo!()
    }

    fn respond_error(
        &self,
        headers: &HeaderMap,
        status: StatusCode,
        message: impl std::fmt::Display,
    ) -> Response {
        // JSON → Json(ErrorResponse), HX-Request → error fragment, else → error page
        todo!()
    }
}
```

Copy the full `RenderService` implementation from an existing controller
such as `optimizations.rs`. It includes embedded assets (`styles.css`,
chart JS) and `full_page_markup` — keep those as-is and only change the
route-specific logic.

## Step 2 — Add handler functions

Follow the handler signature convention from AGENTS.md:

| Parameter | When |
|---|---|
| `State(app): State<Arc<App>>` | Always |
| `Path(id): Path<Uuid>` | Route has `{id}` |
| `NamespacedQuery(params): NamespacedQuery<Q>` | GET with HTMX query params |
| `input: FormOrJson<P>` | POST that accepts form or JSON |
| `Json(payload): Json<P>` | JSON-only POST (API endpoints) |
| `headers: HeaderMap` | Almost always — content negotiation |
| `uri: Uri` | For building relative URLs |

Annotate handlers with `#[axum::debug_handler]` for better compile-time errors.

Example content-negotiated handler:

```rust
#[axum::debug_handler]
async fn list_handler(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
) -> Response {
    let render = RenderService;
    match app.services().some_service().do_something().await {
        Ok(result) => render.respond(&headers, StatusCode::OK, &result),
        Err(e) => render.respond_error(&headers, StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}
```

Example API-only handler:

```rust
#[axum::debug_handler]
async fn create_api_handler(
    State(app): State<Arc<App>>,
    Json(payload): Json<CreatePayload>,
) -> Result<(StatusCode, Json<CreateResponse>), (StatusCode, Json<ErrorResponse>)> {
    match app.services().some_service().create(payload.into()).await {
        Ok(id) => Ok((StatusCode::CREATED, Json(CreateResponse { id }))),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse::new(e.to_string())))),
    }
}
```

## Step 3 — Define the router

```rust
pub fn router(app: Arc<App>) -> ApiRouter {
    ApiRouter::new()
        .api_route(
            "/resource",
            get_with(list_handler, |op| {
                op.operation_id("list_resource")
                    .tag("TagName")
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .api_route(
            "/resource",
            post_with(create_handler, |op| {
                op.operation_id("create_resource")
                    .tag("TagName")
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .api_route(
            "/resource/{id}",
            get_with(get_handler, |op| {
                op.operation_id("get_resource")
                    .tag("TagName")
                    .response_with::<500, Json<ErrorResponse>, _>(|r| {
                        r.description("Internal server error")
                    })
            }),
        )
        .with_state(app)
}
```

Use `operation_id` for unique OpenAPI operation IDs. Use `tag` for grouping
related endpoints. Every route MUST include a 500 error response annotation.

## Step 4 — Register in `controllers/mod.rs`

```rust
pub mod resource;

pub use resource::CreatePayload;
pub use resource::CreateResponse;

pub fn router(app: Arc<App>) -> ApiRouter {
    ApiRouter::new()
        // ...
        .merge(resource::router(app))
}
```

Re-export request and response types so `api_client` can consume them.

## Step 5 — Create view types

Create `src/views/{resource}.rs` with view structs that implement both
`maud::Render` and `serde::Serialize`:

```rust
#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct SomeView {
    id: String,
}

impl SomeView {
    pub fn new(id: String) -> Self {
        Self { id }
    }
}

impl maud::Render for SomeView {
    fn render(&self) -> maud::Markup {
        html! {
            // Full page or partial HTML fragment
        }
    }
}
```

Register the module in `src/views/mod.rs`:

```rust
pub mod resource;
```

## Payload/query struct conventions

- Derive `#[derive(Debug, Deserialize, schemars::JsonSchema)]`
- Use `#[serde(rename_all = "snake_case")]` on enums
- Use `#[serde(default)]` with custom deserializers for non-standard optional fields
- Query fields are `Option<T>` for optional params

## Reference

- See AGENTS.md **Controllers** section for extractor details and patterns
- See AGENTS.md **Views** section for rendering conventions
- See existing controllers (`optimizations.rs`, `requests.rs`) for complete examples
