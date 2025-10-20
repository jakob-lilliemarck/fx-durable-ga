
## Working methodology
1. Always re-read the file prior to mutating its contents! The contents might have changed!
2. Always run `cargo check` before completing a task. We only need to check changed files. Check for:
  - no errors
  - no deprecation warnings
  - no unused imports.
3. Always run `cargo test` before completing a task. We only need to run tests affected by the change.

## Documentation guidelines
- Documentation should be present but should be *brief* and *simple*. No lengthy doc comments or code comments!
- Doc comments `///` are intended for the _client user_. They should document the contract, and clearly outline how the client user may interact with the code and what to expect from it.

**Documentation by visibility level:**
- `fn` (private) - No doc comments needed
- `pub(super)`, `pub(self)`, `pub(crate)` - Simple, brief doc comments describing what the method does, how to call it, and what to expect
- `pub fn` (public API) - More comprehensive doc comments describing what the method does, how to call it, what to expect, include examples for complex method calls. Keep simple and lean.

**Important:** Doc comments should focus on _what the method does_, _how to call it_, and _what to expect_. They should NOT reveal implementation details - those belong in code comments.

- Code comments `//` are intended for the _developer_. They should capture important implementation details, and make the code more approachable while working on it. Especially logic conditions that may be hard for humans to understand.
- After writing documentation, always run doc tests for the modified file to ensure correctness - fix any errors.

## Logging and instrumentation guidelines
- Use tracing `#[instrument(level = "debug")]` annotation to instrument methods. The default level is "info", so we explicitly specify "debug".
- _Most methods_ **should** be instrumented, except trivial methods:
  - Simple getters that don't compute anything
  - Simple constructors that just pass values through
  - Getters that compute values or constructors with logic **should** be instrumented
- Review existing `#[instrument]` annotations to ensure they cover all loggable arguments and use the right log level.
- Add `#[instrument(level = "debug")]` to methods that are missing it.
- Events of special _business concern_ should be logged using tracing events macro `tracing::info!`
- Events that **should never occur** should be logged using tracing events macro `tracing::warn!`
- Errors that are swallowed or handled should be logged using tracing events macro `tracing::error!` - however, do not log errors that are returned to the caller or that are otherwise handled in the code.

## Test guidelines
- Keep tests _simple_ and _readable_
- When writing string uuids in tests, keep them human readable like "00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002", etc.

## Other guidelines
- Use `Uuid::now_v7()`, not `Utc::new_v4()`
