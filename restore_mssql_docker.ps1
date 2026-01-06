param(
  [Parameter(Mandatory=$true)]
  [string]$DB_NAME,

  # internal flag to prevent infinite relaunch
  [switch]$Relaunched
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($msg){ Write-Host "✅ $msg" }
function Warn($msg){ Write-Host "⚠️ $msg" }
function Err($msg){ Write-Host "❌ $msg"; exit 1 }

function HasCmd($name){
  return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function SelfRelaunch {
  param(
    [string]$Reason
  )
  if ($Relaunched) {
    Warn "Relaunch already happened once. Continuing without relaunch. ($Reason)"
    return
  }

  $scriptPath = $MyInvocation.MyCommand.Path
  Info "Self-relaunching script ($Reason)..."
  # Relaunch in a fresh PowerShell process to refresh PATH
  $args = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$scriptPath`"",
    "-DB_NAME", "`"$DB_NAME`"",
    "-Relaunched"
  )

  # Preserve env var SQL_PASS if user supplied it (optional)
  # Not strictly necessary, but nice to keep parity with linux preserve-env
  Start-Process -FilePath "powershell.exe" -ArgumentList $args -Wait -NoNewWindow
  exit 0
}

function Start-DockerDesktopIfPresent {
  $paths = @(
    "$Env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
    "$Env:ProgramFiles(x86)\Docker\Docker\Docker Desktop.exe"
  )
  foreach ($p in $paths) {
    if (Test-Path $p) {
      Info "Starting Docker Desktop..."
      Start-Process -FilePath $p | Out-Null
      return $true
    }
  }
  return $false
}

function Wait-ForDocker([int]$Seconds = 180) {
  Info "Waiting for Docker daemon (up to $Seconds seconds)..."
  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    try {
      docker info | Out-Null
      Info "Docker daemon is ready."
      return $true
    } catch {
      Start-Sleep -Seconds 3
    }
  }
  return $false
}

function Install-DockerDesktop {
  # Prefer winget
  if (HasCmd "winget") {
    Info "Installing Docker Desktop via winget..."
    winget install -e --id Docker.DockerDesktop --accept-source-agreements --accept-package-agreements
    return $true
  }

  # Optional: Chocolatey
  if (HasCmd "choco") {
    Info "Installing Docker Desktop via Chocolatey..."
    choco install docker-desktop -y
    return $true
  }

  return $false
}

function TryAddDockerToPathForSession {
  # Common Docker CLI path after Docker Desktop install
  $candidate = "$Env:ProgramFiles\Docker\Docker\resources\bin\docker.exe"
  if (Test-Path $candidate) {
    $binDir = Split-Path -Parent $candidate
    if ($env:PATH -notlike "*$binDir*") {
      $env:PATH = "$binDir;$env:PATH"
      Info "Added docker CLI to PATH for this session: $binDir"
    }
    return $true
  }
  return $false
}

# -------------------------
# Paths (fixed)
# -------------------------
$ScriptPath  = $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptPath
$BackupDir   = Join-Path $ProjectRoot "data\external"
$EnvPath     = Join-Path $ProjectRoot ".env"

if (!(Test-Path $BackupDir -PathType Container)) {
  Err "Backup directory not found: $BackupDir"
}

# Find .bak files ONLY in that directory
$bakFiles = Get-ChildItem -Path $BackupDir -Filter *.bak -File | Sort-Object Name

if ($bakFiles.Count -eq 0) {
  Err "No .bak file found in: $BackupDir"
}
if ($bakFiles.Count -gt 1) {
  Write-Host "❌ Multiple .bak files found in $BackupDir:"
  $bakFiles | ForEach-Object { Write-Host " - $($_.FullName)" }
  Err "Keep only ONE .bak file in that directory."
}

$BakPath = $bakFiles[0].FullName
Info "Using backup file: $BakPath"

# -------------------------
# SQL creds (prompt like Linux)
# Priority: env var SQL_PASS -> prompt
# -------------------------
$SQL_HOST = "localhost"
$SQL_USER = "sa"

$SQL_PASS = $env:SQL_PASS
if ([string]::IsNullOrWhiteSpace($SQL_PASS)) {
  $secure = Read-Host "SQL Server SA password" -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try { $SQL_PASS = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
  finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}
if ([string]::IsNullOrWhiteSpace($SQL_PASS)) {
  Err "SQL Server SA password is required."
}

# -------------------------
# Ensure Docker Desktop installed + docker available
# -------------------------
$installedNow = $false

if (!(HasCmd "docker")) {
  Warn "Docker CLI not found. Installing Docker Desktop..."
  $ok = Install-DockerDesktop
  if (-not $ok) {
    Err "Could not auto-install Docker Desktop (no winget/choco). Please install manually, then rerun."
  }
  $installedNow = $true

  # docker might not be on PATH in the current session
  TryAddDockerToPathForSession | Out-Null

  # Relaunch to refresh PATH/environment reliably
  SelfRelaunch "Docker Desktop installed"
}

# docker exists, but might be newly installed and not discoverable by current session
# Try to add docker to PATH if docker command still fails oddly
try {
  docker --version | Out-Null
} catch {
  if (TryAddDockerToPathForSession) {
    SelfRelaunch "Docker CLI path fixed"
  }
}

# -------------------------
# Ensure Docker daemon running
# -------------------------
try {
  docker info | Out-Null
} catch {
  Warn "Docker daemon is not running. Attempting to start Docker Desktop..."
  $started = Start-DockerDesktopIfPresent
  if (-not $started) {
    Warn "Docker Desktop executable not found. Please start Docker Desktop manually."
  }

  # If Docker was installed earlier in this run, do one more relaunch after starting Desktop
  if (-not (Wait-ForDocker -Seconds 180)) {
    # On first run, Docker Desktop may need user to accept terms; do one relaunch attempt if we haven't.
    if (-not $Relaunched) {
      SelfRelaunch "Docker daemon not ready (post-start)"
    }

    Err @"
Docker daemon still not ready.

Please:
1) Open Docker Desktop once (Start Menu -> Docker Desktop)
2) Finish initial setup / accept terms if prompted
3) Then rerun:
   powershell -ExecutionPolicy Bypass -File .\restore_mssql_docker.ps1 -DB_NAME $DB_NAME
"@
  }
}

# -------------------------
# SQL Server container config
# -------------------------
$CONTAINER_NAME = "mssql_restore"
$MSSQL_IMAGE    = "mcr.microsoft.com/mssql/server:2022-latest"
$MSSQL_PORT     = 14333  # host port

# Remove existing container if any
$existing = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $CONTAINER_NAME }
if ($existing) {
  Info "Removing existing container..."
  docker rm -f $CONTAINER_NAME | Out-Null
}

Info "Starting SQL Server container..."
docker run -d `
  --name $CONTAINER_NAME `
  -e "ACCEPT_EULA=Y" `
  -e "MSSQL_SA_PASSWORD=$SQL_PASS" `
  -p "$MSSQL_PORT`:1433" `
  $MSSQL_IMAGE | Out-Null

# -------------------------
# Wait for SQL Server readiness
# -------------------------
Info "Waiting for SQL Server to be ready..."
$ready = $false
for ($i=1; $i -le 60; $i++) {
  try {
    docker exec $CONTAINER_NAME /opt/mssql-tools18/bin/sqlcmd `
      -C -S $SQL_HOST -U $SQL_USER -P $SQL_PASS -Q "SELECT 1" | Out-Null
    $ready = $true
    break
  } catch {
    Start-Sleep -Seconds 2
  }
}
if (-not $ready) { Err "SQL Server did not start in time." }
Info "SQL Server is ready."

# -------------------------
# Copy backup into container
# -------------------------
$IN_CONTAINER_DIR = "/var/opt/mssql/backup"
$bakName = Split-Path -Leaf $BakPath
$IN_CONTAINER_BAK = "$IN_CONTAINER_DIR/$bakName"

docker exec $CONTAINER_NAME mkdir -p $IN_CONTAINER_DIR | Out-Null
docker cp $BakPath "$CONTAINER_NAME`:$IN_CONTAINER_BAK" | Out-Null
Info "Backup copied to container: $IN_CONTAINER_BAK"

# -------------------------
# Detect logical names
# -------------------------
Info "Detecting logical names..."
$filelist = docker exec $CONTAINER_NAME /opt/mssql-tools18/bin/sqlcmd `
  -C -S $SQL_HOST -U $SQL_USER -P $SQL_PASS `
  -Q "RESTORE FILELISTONLY FROM DISK = N'$IN_CONTAINER_BAK';" -s "|" -W

$dataLogical = $null
$logLogical  = $null

foreach ($line in ($filelist -split "`n")) {
  $t = $line.Trim()
  if ($t.Length -eq 0) { continue }
  if (-not $dataLogical -and $t -match "\|D\|") { $dataLogical = ($t -split "\|")[0].Trim() }
  if (-not $logLogical  -and $t -match "\|L\|") { $logLogical  = ($t -split "\|")[0].Trim() }
}

if (-not $dataLogical -or -not $logLogical) {
  Write-Host $filelist
  Err "Logical names not found."
}

Info "Data logical name: $dataLogical"
Info "Log logical name:  $logLogical"

# -------------------------
# Restore (WITH REPLACE)
# -------------------------
$MDF = "/var/opt/mssql/data/$DB_NAME.mdf"
$LDF = "/var/opt/mssql/data/${DB_NAME}_log.ldf"

Info "Restoring database '$DB_NAME'..."
$restoreQuery = @"
IF DB_ID(N'$DB_NAME') IS NOT NULL
BEGIN
  ALTER DATABASE [$DB_NAME] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
END
RESTORE DATABASE [$DB_NAME]
FROM DISK = N'$IN_CONTAINER_BAK'
WITH
  MOVE N'$dataLogical' TO N'$MDF',
  MOVE N'$logLogical' TO N'$LDF',
  REPLACE,
  RECOVERY;
ALTER DATABASE [$DB_NAME] SET MULTI_USER;
"@

docker exec $CONTAINER_NAME /opt/mssql-tools18/bin/sqlcmd `
  -C -S $SQL_HOST -U $SQL_USER -P $SQL_PASS -Q $restoreQuery | Out-Host

# -------------------------
# .env helpers (update + remove password)
# -------------------------
function Ensure-EnvFile {
  if (!(Test-Path $EnvPath)) { New-Item -Path $EnvPath -ItemType File -Force | Out-Null }
}
function Set-EnvVar([string]$key, [string]$value) {
  Ensure-EnvFile
  $lines = Get-Content $EnvPath -ErrorAction SilentlyContinue
  $found = $false
  $out = New-Object System.Collections.Generic.List[string]
  foreach ($line in $lines) {
    if ($line -match "^\Q$key\E=") { $out.Add("$key=$value"); $found = $true }
    else { $out.Add($line) }
  }
  if (-not $found) { $out.Add("$key=$value") }
  Set-Content -Path $EnvPath -Value $out -Encoding UTF8
}
function Remove-EnvVar([string]$key) {
  if (!(Test-Path $EnvPath)) { return }
  $lines = Get-Content $EnvPath -ErrorAction SilentlyContinue
  $out = $lines | Where-Object { $_ -notmatch "^\Q$key\E=" }
  Set-Content -Path $EnvPath -Value $out -Encoding UTF8
}

Info "Updating .env with SQL connection settings..."
Set-EnvVar "PHARMACYSTORE_SQL_SERVER" "localhost,$MSSQL_PORT"
Set-EnvVar "PHARMACYSTORE_SQL_DATABASE" $DB_NAME
Set-EnvVar "PHARMACYSTORE_SQL_USERNAME" $SQL_USER
Remove-EnvVar "PHARMACYSTORE_SQL_PASSWORD"
Set-EnvVar "PHARMACYSTORE_SQL_DRIVER" "ODBC Driver 18 for SQL Server"
Set-EnvVar "PHARMACYSTORE_SQL_ENCRYPT" "false"
Set-EnvVar "PHARMACYSTORE_SQL_TRUST_CERT" "true"

Info "🎉 DONE — Database restored: $DB_NAME"
Info "Connect: localhost:$MSSQL_PORT | user=sa"
