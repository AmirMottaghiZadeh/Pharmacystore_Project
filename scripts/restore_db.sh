#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DB_NAME="${1:-PharmacyStore}"

exec bash "$PROJECT_ROOT/restore_mssql_docker.sh" "$DB_NAME"
