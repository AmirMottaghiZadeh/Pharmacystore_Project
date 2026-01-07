import hashlib
import random
import re

import numpy as np

from pharmacystore import run_utils


def test_set_global_seed_deterministic() -> None:
    run_utils.set_global_seed(123)
    first = (random.random(), np.random.rand())
    run_utils.set_global_seed(123)
    second = (random.random(), np.random.rand())
    assert first == second


def test_timestamp_run_id_format() -> None:
    value = run_utils.timestamp_run_id()
    assert re.fullmatch(r"\d{8}_\d{6}", value)


def test_sanitize_run_tag_removes_invalid() -> None:
    assert run_utils.sanitize_run_tag("  My Tag!! ") == "MyTag"
    assert run_utils.sanitize_run_tag("---bad__") == "bad"


def test_create_run_dir_uses_sanitized_tag(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_utils, "timestamp_run_id", lambda: "20240101_010101")
    run_dir = run_utils.create_run_dir(tmp_path, " My Tag!! ")
    assert run_dir.exists()
    assert run_dir.name == "20240101_010101_MyTag"


def test_write_json_and_compute_file_hash(tmp_path) -> None:
    payload = {"b": 2, "a": 1}
    path = tmp_path / "data" / "payload.json"
    run_utils.write_json(path, payload)
    text = path.read_text(encoding="utf-8")
    assert "\"a\": 1" in text
    expected_md5 = hashlib.md5(path.read_bytes()).hexdigest()
    assert run_utils.compute_file_hash(path) == expected_md5
    expected_sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
    assert run_utils.compute_file_hash(path, algo="sha1") == expected_sha1
