from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PHARMACYSTORE_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    sql_server: str | None = Field(default=None)
    sql_database: str | None = Field(default=None)
    sql_username: str | None = Field(default=None)
    sql_password: str | None = Field(default=None)
    sql_driver: str = Field(default="ODBC Driver 18 for SQL Server")
    sql_encrypt: bool = Field(default=False)
    sql_trust_cert: bool = Field(default=True)

    data_dir: str = Field(default="data")
    artifacts_dir: str = Field(default="artifacts")

    random_seed: int = Field(default=42)
    run_walk_forward: bool = Field(default=True)

    def data_path(self) -> Path:
        return Path(self.data_dir)

    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
