#!/bin/bash

# Script to truncate all tables in the fx-durable-ga project schemas
# This clears all data but keeps the table structure intact

set -e  # Exit on error
set -o pipefail  # Exit on pipe failures

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL environment variable is not set"
    exit 1
fi

echo "🧹 Truncating all tables in fx-durable-ga project schemas..."

# Function to truncate tables in a schema
truncate_schema_tables() {
    local schema="$1"
    shift
    local tables=("$@")
    
    echo "📋 Truncating tables in schema: $schema"
    
    for table in "${tables[@]}"; do
        echo "  - Truncating $schema.$table"
        if ! psql "$DATABASE_URL" -c "TRUNCATE TABLE $schema.\"$table\" CASCADE;" >/dev/null 2>&1; then
            echo "    ❌ Failed to truncate $schema.$table"
            return 1
        fi
    done
    echo "    ✅ Completed $schema"
}

# fx_durable_ga schema tables
FX_DURABLE_GA_TABLES=(
    "fitness"
    "genotypes"
    "requests"
    "morphologies"
)

# fx_event_bus schema tables (excluding _sqlx_migrations)
FX_EVENT_BUS_TABLES=(
    "attempts_dead"
    "attempts_failed"
    "attempts_succeeded"
    "events_acknowledged"
    "events_unacknowledged"
)

# fx_mq_jobs schema tables (excluding _sqlx_migrations)
FX_MQ_JOBS_TABLES=(
    "attempts_dead"
    "attempts_failed"
    "attempts_succeeded"
    "errors"
    "leases"
    "messages_attempted"
    "messages_unattempted"
)

echo ""
# Truncate tables in order (most dependent first)
if truncate_schema_tables "fx_durable_ga" "${FX_DURABLE_GA_TABLES[@]}" && \
   truncate_schema_tables "fx_event_bus" "${FX_EVENT_BUS_TABLES[@]}" && \
   truncate_schema_tables "fx_mq_jobs" "${FX_MQ_JOBS_TABLES[@]}"; then
    echo ""
    echo "✅ Successfully truncated all tables!"
else
    echo ""
    echo "❌ Failed to truncate some tables!"
    exit 1
fi

echo ""
echo "📊 Verifying truncation with sample table counts:"

# Simple verification with direct table counts
for schema_table in "fx_durable_ga.genotypes" "fx_event_bus.events_acknowledged" "fx_mq_jobs.messages_attempted"; do
    count=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM $schema_table;" 2>/dev/null | tr -d ' ')
    if [ "$count" = "0" ]; then
        echo "  ✅ $schema_table: $count rows"
    else
        echo "  ⚠️  $schema_table: $count rows (expected 0)"
    fi
done

echo ""
echo "🎉 Truncation complete! Ready for fresh testing."
