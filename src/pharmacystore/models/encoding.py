"""Feature preparation and categorical encoding."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare lag-safe features for XGBoost; return X (unencoded), y, and meta."""
    data = df.copy().sort_values(["DrugId", "week_start"]).reset_index(drop=True)
    data["week_start_ts"] = pd.to_datetime(data["week_start"]).astype("int64") // 10**9

    if "UPW_lag1" not in data.columns:
        data["UPW_lag1"] = data.groupby("DrugId")["UPW"].shift(1)

    lag_series = data["UPW_lag1"]
    if "UPW_rollmean_4" not in data.columns:
        data["UPW_rollmean_4"] = lag_series.rolling(window=4, min_periods=4).mean()
    if "UPW_rollmean_8" not in data.columns:
        data["UPW_rollmean_8"] = lag_series.rolling(window=8, min_periods=8).mean()
    if "UPW_rollmean_12" not in data.columns:
        data["UPW_rollmean_12"] = lag_series.rolling(window=12, min_periods=12).mean()
    if "UPW_rollmax_8" not in data.columns:
        data["UPW_rollmax_8"] = lag_series.rolling(window=8, min_periods=8).max()
    if "UPW_rollstd_8" not in data.columns:
        data["UPW_rollstd_8"] = lag_series.rolling(window=8, min_periods=8).std()
    if "weeks_since_peak_13" not in data.columns:
        data["weeks_since_peak_13"] = (
            lag_series.rolling(window=13, min_periods=4)
            .apply(lambda s: len(s) - 1 - int(np.argmax(s.values)))
        )

    data["spike_score_8"] = (lag_series - data["UPW_rollmean_8"]) / (data["UPW_rollstd_8"] + 1e-6)

    data = data.replace([np.inf, -np.inf], np.nan)
    history_cols = [
        col
        for col in [
            "UPW_lag1",
            "UPW_lag2",
            "UPW_lag4",
            "UPW_rollmean_4",
            "UPW_rollmean_8",
            "UPW_rollmean_12",
            "UPW_rollmax_8",
            "UPW_rollstd_8",
            "weeks_since_peak_13",
        ]
        if col in data.columns
    ]
    if history_cols:
        data = data.dropna(subset=history_cols).reset_index(drop=True)

    meta_cols = [
        col
        for col in [
            "week_start",
            "week_start_ts",
            "DrugId",
            "genericname",
            "brandname",
            "saleCategory",
            "priceCategory",
            "UPW_rollmean_8",
            "UPW_rollstd_8",
            "UPW_rollmax_8",
            "weeks_since_peak_13",
            "spike_score_8",
            "quarter",
        ]
        if col in data.columns
    ]
    meta = data[meta_cols].copy()

    y = data["UPW"].astype(float)
    drop_target_like = [
        "UPW",
        "UniquePackets",
        "week_start",
        "SalesCategory",
        "saleCategory",
        "PriceCategory",
        "priceCategory",
        "Scale",
        "PriceScale",
        "diff_rate",
        "Z_score",
        "trend_short_vs_long",
        "trend_slope_8w",
        "trend_accel",
        "trend_slope_8w_norm",
        "pct_trend_8w",
        "seasonality_strength_8w",
        "regime_prob_8w",
        "week_sin",
        "week_cos",
        "spike_flag",
        "spike_score_8",
    ]
    X_full = data.drop(columns=[c for c in drop_target_like if c in data.columns])
    safe_features = [
        "UPW_lag1",
        "UPW_lag2",
        "UPW_lag4",
        "UPW_rollmean_4",
        "UPW_rollmean_8",
        "UPW_rollmean_12",
        "UPW_rollmax_8",
        "UPW_rollstd_8",
        "weeks_since_peak_13",
        "week_of_year",
        "month",
        "official_holiday_days",
        "quarter",
    ]
    categorical_cols = [
        c
        for c in X_full.columns
        if X_full[c].dtype == "object" or str(X_full[c].dtype).startswith("category")
    ]
    keep_cols = [c for c in safe_features if c in X_full.columns] + categorical_cols
    X = X_full[keep_cols].copy()

    return X, y, meta


def encode_categoricals(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    """Fit label encoders on train only and transform valid with an UNK bucket."""
    X_train_enc = X_train.copy()
    X_valid_enc = X_valid.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in X_train_enc.select_dtypes(include=["object", "category"]).columns:
        train_vals = X_train_enc[col].fillna("NA").astype(str)
        valid_vals = X_valid_enc[col].fillna("NA").astype(str)

        le = LabelEncoder()
        le.fit(train_vals)
        if "UNK" not in le.classes_:
            le.classes_ = np.append(le.classes_, "UNK")
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}

        X_train_enc[col] = train_vals.map(mapping).astype(int)
        X_valid_enc[col] = valid_vals.map(lambda v: mapping.get(v, mapping["UNK"])).astype(int)
        encoders[col] = le

    X_train_enc = X_train_enc.fillna(0)
    X_valid_enc = X_valid_enc.fillna(0)
    return X_train_enc, X_valid_enc, encoders


def apply_label_encoders(
    X: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    """Transform categoricals with fitted encoders; unseen values map to UNK."""
    X_enc = X.copy()
    for col, le in encoders.items():
        if col not in X_enc.columns:
            continue
        vals = X_enc[col].fillna("NA").astype(str)
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        X_enc[col] = vals.map(lambda v: mapping.get(v, mapping["UNK"])).astype(int)
    X_enc = X_enc.fillna(0)
    return X_enc


def select_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pick stable, low-variance features for the baseline stage."""
    preferred = [
        "UPW_lag1",
        "UPW_lag2",
        "UPW_lag4",
        "UPW_rollmean_4",
        "UPW_rollmean_8",
        "UPW_rollmean_12",
        "UPW_rollmax_8",
        "UPW_rollstd_8",
        "weeks_since_peak_13",
        "week_of_year",
        "month",
        "official_holiday_days",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[cols].copy().fillna(0)
