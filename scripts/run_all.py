from __future__ import annotations

import getpass
import os
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
ATC_REFERENCE = PROJECT_ROOT / "data" / "external" / "who_atc_ddd.csv"
RESTORE_SCRIPT = PROJECT_ROOT / "scripts" / "restore_db.sh"
RESTORE_PS_SCRIPT = PROJECT_ROOT / "restore_mssql_docker.ps1"


def _prompt_yes_no(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        choice = input(f"{label} [{suffix}]: ").strip().lower()
        if not choice:
            return default
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _prompt_secret(label: str) -> str:
    while True:
        value = getpass.getpass(f"{label}: ").strip()
        if value:
            return value
        print("Value is required.")


def _run_restore(env: dict[str, str]) -> None:
    system = platform.system().lower()
    if system.startswith("windows"):
        subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(RESTORE_PS_SCRIPT),
                "-DB_NAME",
                "PharmacyStore",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )
        return
    if system == "linux":
        subprocess.run(["bash", str(RESTORE_SCRIPT)], cwd=PROJECT_ROOT, env=env, check=True)
        return
    raise RuntimeError(f"Unsupported operating system: {platform.system()}")


def _install_dependencies() -> None:
    if not REQUIREMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing {REQUIREMENTS_PATH}")

    editable = _prompt_yes_no(
        "Install project in editable mode (recommended)", default=True
    )
    if editable:
        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _ensure_required_files() -> None:
    if not ATC_REFERENCE.exists():
        raise FileNotFoundError(
            "Missing ATC reference file: "
            f"{ATC_REFERENCE}. Place it and rerun this script."
        )


def _load_env_vars(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _ensure_db_settings(env: dict[str, str]) -> None:
    env_path = PROJECT_ROOT / ".env"
    env_values = _load_env_vars(env_path)
    required = [
        "PHARMACYSTORE_SQL_SERVER",
        "PHARMACYSTORE_SQL_DATABASE",
        "PHARMACYSTORE_SQL_USERNAME",
    ]
    missing = [key for key in required if not (env.get(key) or env_values.get(key))]
    if missing:
        raise RuntimeError(
            "Missing DB settings in .env: "
            + ", ".join(missing)
            + ". Run restore_mssql_docker.sh to restore the backup and write .env."
        )
    if not (env.get("PHARMACYSTORE_SQL_PASSWORD") or env_values.get("PHARMACYSTORE_SQL_PASSWORD")):
        raise RuntimeError(
            "Missing DB password. Provide it when prompted or set PHARMACYSTORE_SQL_PASSWORD."
        )


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path
    return env


def main() -> int:
    print("This script installs dependencies and runs the full pipeline.")
    sql_password = _prompt_secret("SQL password")
    env = _build_env()
    env["SQL_PASS"] = sql_password
    env["PHARMACYSTORE_SQL_PASSWORD"] = sql_password
    _run_restore(env)
    _ensure_required_files()
    _install_dependencies()
    _ensure_db_settings(env)
    subprocess.run(
        [sys.executable, "-m", "pharmacystore.pipeline", "full"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
