# fx-durable-ga

Durable GA event driven optimization loop on PostgreSQL

## Development Setup

### Database Setup

1. Set up your database connection in `.env.local`
2. If you encounter SQLx compile-time errors about missing relations after rebuilding:

```bash
# Load environment variables and set search path for all schemas
export $(cat .env.local | xargs)
export DATABASE_URL="${DATABASE_URL}?options=-csearch_path=fx_mq_jobs,fx_event_bus,fx_durable_ga,public"

# Generate SQLx query cache
cargo sqlx prepare
```

This creates the `.sqlx/` directory with cached query metadata, allowing compilation without a live database connection.

### Running Examples

```bash
cargo run --example cube
```

### Test coverage
```sh
cargo llvm-cov --html --output-dir coverage -- models::
```
