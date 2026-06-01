#!/usr/bin/env bash
set -euo pipefail

DUMP_DIR="./db_dumps"

if [[ -z "${DATABASE_URL_REMOTE:-}" ]]; then
  echo "[db_backup] ERROR: DATABASE_URL_REMOTE is not set"
  exit 1
fi

mkdir -p "$DUMP_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
DUMP_FILE="${DUMP_DIR}/db_backup_${STAMP}.dump"

echo "[db_backup] Starting backup"
echo "[db_backup] Source: DATABASE_URL_REMOTE"
echo "[db_backup] Output: ${DUMP_FILE}"

PGCONNECT_TIMEOUT=5 pg_dump -Fc --verbose "$DATABASE_URL_REMOTE" -f "$DUMP_FILE"

echo "[db_backup] Backup complete"
