/// Query extractor that strips namespace prefixes from query parameters before
/// deserializing. Parameters are namespaced by the frontend using "--" as a
/// separator (e.g. `optimization-list-items--search=foo`). The extractor splits
/// each key at "--" using `rsplit_once` and uses the last segment, so both
/// namespaced and plain parameters work transparently:
///
/// - `optimization-list-items--search=foo` → `search=foo`
/// - `search=foo` → `search=foo` (unchanged)
///
/// This allows the frontend to namespace parameters per HTMX target element
/// without the backend needing any knowledge of the namespacing scheme.
pub struct NamespacedQuery<T>(pub T);

const NS_SEPARATOR: &str = "--";

impl<T, S> axum::extract::FromRequestParts<S> for NamespacedQuery<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = axum::extract::rejection::QueryRejection;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        let stripped_query = parts
            .uri
            .query()
            .unwrap_or_default()
            .split('&')
            .filter(|s| !s.is_empty())
            .map(|pair| {
                let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
                let stripped_key = k
                    .rsplit_once(NS_SEPARATOR)
                    .map(|(_, suffix)| suffix)
                    .unwrap_or(k);
                format!("{}={}", stripped_key, v)
            })
            .collect::<Vec<_>>()
            .join("&");

        let new_uri = axum::http::Uri::builder()
            .path_and_query(format!("{}?{}", parts.uri.path(), stripped_query))
            .build()
            .expect("failed to build URI from stripped query string");

        parts.uri = new_uri;

        let axum::extract::Query(value) =
            axum::extract::Query::<T>::from_request_parts(parts, state).await?;

        Ok(NamespacedQuery(value))
    }
}

impl<T> aide::OperationInput for NamespacedQuery<T>
where
    axum::extract::Query<T>: aide::OperationInput,
{
    fn operation_input(
        ctx: &mut aide::generate::GenContext,
        operation: &mut aide::openapi::Operation,
    ) {
        axum::extract::Query::<T>::operation_input(ctx, operation);
    }
}
