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
    
    # Model hyperparameters
    xgb_eta: float = Field(default=0.05)
    xgb_max_depth: int = Field(default=5)
    xgb_subsample: float = Field(default=0.75)
    xgb_colsample_bytree: float = Field(default=0.75)
    xgb_min_child_weight: float = Field(default=8.0)
    xgb_gamma: float = Field(default=0.4)
    xgb_reg_lambda: float = Field(default=1.5)
    xgb_reg_alpha: float = Field(default=0.1)
    xgb_num_boost_round: int = Field(default=250)
    xgb_early_stopping_rounds: int = Field(default=35)
    
    # Training configuration
    train_frac: float = Field(default=0.7)
    valid_frac: float = Field(default=0.15)
    cv_n_splits: int = Field(default=3)

    def data_path(self) -> Path:
        return Path(self.data_dir)

    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
