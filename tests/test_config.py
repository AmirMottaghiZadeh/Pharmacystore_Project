from pathlib import Path

from pharmacystore import config


def test_get_settings_cache_and_paths(monkeypatch) -> None:
    config.get_settings.cache_clear()
    monkeypatch.setenv("PHARMACYSTORE_DATA_DIR", "data-test")
    monkeypatch.setenv("PHARMACYSTORE_ARTIFACTS_DIR", "artifacts-test")

    settings = config.get_settings()
    assert settings.data_path() == Path("data-test")
    assert settings.artifacts_path() == Path("artifacts-test")

    again = config.get_settings()
    assert settings is again
    config.get_settings.cache_clear()
