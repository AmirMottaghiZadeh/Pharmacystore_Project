#!/usr/bin/env bash
set -euo pipefail

# =========================
# Self elevate (sudo) + chmod
# =========================
SCRIPT_PATH="$(readlink -f "$0")"
[[ -x "$SCRIPT_PATH" ]] || chmod +x "$SCRIPT_PATH"
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "🔐 Re-running with sudo..."
  exec sudo --preserve-env=SQL_PASS bash "$SCRIPT_PATH" "$@"
fi

info(){ echo "✅ $*"; }
warn(){ echo "⚠️ $*"; }
err(){ echo "❌ $*" >&2; exit 1; }

escape_sed(){ printf '%s' "$1" | sed -e 's/[&/]/\\&/g'; }
update_env_var(){
  local key="$1"
  local value="$2"
  local escaped
  escaped="$(escape_sed "$value")"
  if [[ -f "$ENV_PATH" ]] && grep -q "^${key}=" "$ENV_PATH"; then
    sed -i "s|^${key}=.*|${key}=${escaped}|" "$ENV_PATH"
  else
    echo "${key}=${value}" >> "$ENV_PATH"
  fi
}
remove_env_var(){
  local key="$1"
  [[ -f "$ENV_PATH" ]] || return 0
  sed -i "/^${key}=.*/d" "$ENV_PATH"
}

# -------------------------
# Fix known broken repos that break apt update
# -------------------------
disable_repo_if_matches () {
  local pattern="$1"
  local files=""
  files="$(grep -RIl --exclude-dir=trusted.gpg.d --exclude="*.disabled" \
    "$pattern" /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null || true)"

  if [[ -n "$files" ]]; then
    warn "Disabling repo(s) matching: $pattern"
    echo "$files" | while IFS= read -r f; do
      [[ -z "$f" ]] && continue
      if [[ "$f" == "/etc/apt/sources.list" ]]; then
        sed -i "s|^\(.*$pattern.*\)$|# disabled by restore script: \1|g" /etc/apt/sources.list
      else
        mv "$f" "$f.disabled" 2>/dev/null || true
      fi
    done
  fi
}

# Disable Microsoft trixie repo (404)
disable_repo_if_matches "packages.microsoft.com/debian/trixie"
# Disable hotspotshield repo (SSL issues)
disable_repo_if_matches "repo.hotspotshield.com"
# Disable v2raya repo (DNS issues)
disable_repo_if_matches "apt.v2raya.org"

# =========================
# Fixed SQL Server config
# =========================
SQL_HOST="localhost"
SQL_USER="sa"
SQL_PASS="${SQL_PASS:-}"
if [[ -z "$SQL_PASS" ]]; then
  read -r -s -p "SQL Server SA password: " SQL_PASS
  echo
fi
[[ -n "$SQL_PASS" ]] || err "SQL Server SA password is required."

CONTAINER_NAME="mssql_restore"
MSSQL_IMAGE="mcr.microsoft.com/mssql/server:2022-latest"
MSSQL_PORT="14333"     # host port (external) - can be 14333 since 1433 is taken on your machine

# =========================
# Inputs
# =========================
DB_NAME="${1:-}"
if [[ -z "$DB_NAME" ]]; then
  err "Usage: bash $0 DB_NAME"
fi

# =========================
# ثابت: فقط از مسیر ./data/external داخل ریشه پروژه
# =========================
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/data/external"
ENV_PATH="${PROJECT_ROOT}/.env"

[[ -d "$BACKUP_DIR" ]] || err "Backup directory not found: $BACKUP_DIR"

# پیدا کردن فایل‌های bak فقط در همین دایرکتوری (نه زیرپوشه‌ها)
mapfile -t BAK_FILES < <(find "$BACKUP_DIR" -maxdepth 1 -type f -iname "*.bak" | sort)

if [[ "${#BAK_FILES[@]}" -eq 0 ]]; then
  err "No .bak file found in: $BACKUP_DIR"
elif [[ "${#BAK_FILES[@]}" -gt 1 ]]; then
  printf "❌ Multiple .bak files found in %s:\n" "$BACKUP_DIR" >&2
  printf " - %s\n" "${BAK_FILES[@]}" >&2
  err "Keep only ONE .bak file in that directory."
fi

BAK_PATH="${BAK_FILES[0]}"
info "Using backup file: $BAK_PATH"

# =========================
# OS info (informational)
# =========================
if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  info "Detected OS: ${PRETTY_NAME}"
fi

# =========================
# Install Docker if missing
# =========================
if ! command -v docker >/dev/null 2>&1; then
  info "Docker not found. Installing Docker..."

  apt-get update -y
  apt-get install -y ca-certificates curl gnupg lsb-release

  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/debian bookworm stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  systemctl enable docker
  systemctl start docker

  info "Docker installed successfully."
else
  info "Docker already installed."
fi

# =========================
# Check Docker daemon
# =========================
docker info >/dev/null 2>&1 || err "Docker daemon not running."

# =========================
# Recreate container
# =========================
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  info "Removing existing container..."
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

info "Starting SQL Server container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  -e "ACCEPT_EULA=Y" \
  -e "MSSQL_SA_PASSWORD=$SQL_PASS" \
  -p "${MSSQL_PORT}:1433" \
  "$MSSQL_IMAGE" >/dev/null

# =========================
# Wait for SQL Server
# =========================
info "Waiting for SQL Server to be ready..."
for i in {1..60}; do
  if docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
      -C -S "$SQL_HOST" -U "$SQL_USER" -P "$SQL_PASS" -Q "SELECT 1" \
      >/dev/null 2>&1; then
    info "SQL Server is ready."
    break
  fi
  sleep 2
  [[ "$i" -eq 60 ]] && err "SQL Server did not start in time."
done

# =========================
# Copy backup
# =========================
IN_CONTAINER_DIR="/var/opt/mssql/backup"
IN_CONTAINER_BAK="$IN_CONTAINER_DIR/$(basename "$BAK_PATH")"

docker exec "$CONTAINER_NAME" mkdir -p "$IN_CONTAINER_DIR"
docker cp "$BAK_PATH" "$CONTAINER_NAME:$IN_CONTAINER_BAK"

# =========================
# Detect logical names
# =========================
info "Detecting logical names..."
FILELIST="$(docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
  -C -S "$SQL_HOST" -U "$SQL_USER" -P "$SQL_PASS" \
  -Q "RESTORE FILELISTONLY FROM DISK = N'$IN_CONTAINER_BAK';" -s "|" -W)"

DATA_LOGICAL="$(echo "$FILELIST" | awk -F"|" 'NR>2 && $0~/\|D\|/ {print $1; exit}')"
LOG_LOGICAL="$(echo "$FILELIST"  | awk -F"|" 'NR>2 && $0~/\|L\|/ {print $1; exit}')"

[[ -n "$DATA_LOGICAL" && -n "$LOG_LOGICAL" ]] || { echo "$FILELIST"; err "Logical names not found."; }

# =========================
# Restore (REPLACE)
# =========================
MDF="/var/opt/mssql/data/${DB_NAME}.mdf"
LDF="/var/opt/mssql/data/${DB_NAME}_log.ldf"

info "Restoring database '$DB_NAME'..."
docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
  -C -S "$SQL_HOST" -U "$SQL_USER" -P "$SQL_PASS" -Q "
IF DB_ID(N'$DB_NAME') IS NOT NULL
BEGIN
  ALTER DATABASE [$DB_NAME] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
END
RESTORE DATABASE [$DB_NAME]
FROM DISK = N'$IN_CONTAINER_BAK'
WITH
  MOVE N'$DATA_LOGICAL' TO N'$MDF',
  MOVE N'$LOG_LOGICAL' TO N'$LDF',
  REPLACE,
  RECOVERY;
ALTER DATABASE [$DB_NAME] SET MULTI_USER;
"

info "🎉 DONE — Database restored: $DB_NAME"
info "Updating .env with SQL connection settings..."
update_env_var "PHARMACYSTORE_SQL_SERVER" "localhost,${MSSQL_PORT}"
update_env_var "PHARMACYSTORE_SQL_DATABASE" "$DB_NAME"
update_env_var "PHARMACYSTORE_SQL_USERNAME" "$SQL_USER"
remove_env_var "PHARMACYSTORE_SQL_PASSWORD"
update_env_var "PHARMACYSTORE_SQL_DRIVER" "ODBC Driver 18 for SQL Server"
update_env_var "PHARMACYSTORE_SQL_ENCRYPT" "false"
update_env_var "PHARMACYSTORE_SQL_TRUST_CERT" "true"
if [[ -n "${SUDO_USER:-}" ]]; then
  chown "${SUDO_USER}:${SUDO_USER}" "$ENV_PATH" || true
fi
info "Connect: localhost:$MSSQL_PORT | user=sa"
