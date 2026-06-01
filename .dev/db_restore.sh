#!/usr/bin/env bash
set -euo pipefail

DUMP_DIR="./db_dumps"

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "[db_restore] ERROR: DATABASE_URL is not set"
  exit 1
fi

HOSTPORT="${DATABASE_URL#*@}"
HOSTPORT="${HOSTPORT%%/*}"
HOST="${HOSTPORT%%:*}"
case "$HOST" in
  localhost|127.0.0.1|0.0.0.0|host.docker.internal) ;;
  *)
    echo "[db_restore] ERROR: DATABASE_URL host is not local (${HOST})"
    exit 1
    ;;
esac

mkdir -p "$DUMP_DIR"

if [[ $# -ge 1 ]]; then
  DUMP_FILE="$1"
else
  DUMP_FILE=$(ls -t "$DUMP_DIR"/db_backup_*.dump 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${DUMP_FILE:-}" ]]; then
  echo "[db_restore] ERROR: No dump files found in ${DUMP_DIR}"
  exit 1
fi

if [[ ! -f "$DUMP_FILE" ]]; then
  echo "[db_restore] ERROR: Dump file not found: $DUMP_FILE"
  exit 1
fi

echo "[db_restore] Dropping local database"
sqlx database drop

echo "[db_restore] Creating local database"
sqlx database create

echo "[db_restore] Restoring from $DUMP_FILE"
pg_restore --no-owner --no-privileges -d "$DATABASE_URL" "$DUMP_FILE"

echo "[db_restore] Restore complete"
