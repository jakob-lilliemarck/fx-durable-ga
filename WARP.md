
## Working methodology
1. Always re-read the file prior to mutating its contents! The contents might have changed!
2. Always run `cargo check` before completing a task. We only need to check changed files. Check for:
  - no errors
  - no deprecation warnings
  - no unused imports.
3. Always run `cargo test` before completing a task. We only need to run tests affected by the change.

## Documentation guidelines
- Documentation should be present but should be *brief*. No lenghty doc comments or code comments!
- Doc comments `///` are intended for the _client user_. They should document the contract, and clearly outline how the client user may interact with the code and what to expectr from it.
- Code comments `//` are intended for the _developer_. They should capture important implementation details, and make the code more approachable while working on it. Especially logic conditions that may be hard for humans to understand.

## Test guidelines
- Keep tests _simple_ and _readable_
- When writing string uuids in tests, keep them human readable like "00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002", etc.

## Other guidelines
- Use `Uuid::now_v7()`, not `Utc::new_v4()`
