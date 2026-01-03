from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def timestamp_run_id() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def sanitize_run_tag(tag: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    cleaned = "".join(ch for ch in tag if ch in allowed)
    return cleaned.strip("-_")


def create_run_dir(artifacts_dir: Path, run_tag: str | None = None) -> Path:
    run_id = timestamp_run_id()
    if run_tag:
        cleaned = sanitize_run_tag(run_tag)
        if cleaned:
            run_id = f"{run_id}_{cleaned}"
    run_dir = artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def compute_file_hash(path: Path, algo: str = "md5") -> str:
    digest = hashlib.new(algo)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
