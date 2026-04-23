"""Train an XGBoost model to forecast weekly UPW using all engineered features."""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from pharmacystore.config import Settings, get_settings
from pharmacystore.pipeline import run_pipeline
from pharmacystore.run_utils import compute_file_hash, create_run_dir, set_global_seed, write_json


ENABLE_SPIKE_POLICY = False


def build_dataset(settings: Settings | None = None) -> pd.DataFrame:
    """Build the full weekly feature set via the existing pipeline."""
    weekly_df = run_pipeline(settings=settings)
    return weekly_df


def _safe_settings_snapshot(settings: Settings) -> dict:
    snapshot = settings.model_dump()
    password = snapshot.pop("sql_password", None)
    snapshot["sql_password_set"] = bool(password)
    return snapshot


def _iso_range(series: pd.Series) -> dict[str, str | None]:
    if series is None:
        return {"min": None, "max": None}
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "max": None}
    return {
        "min": values.min().date().isoformat(),
        "max": values.max().date().isoformat(),
    }


def _ensure_week_start_ts(meta: pd.DataFrame) -> pd.DataFrame:
    if "week_start_ts" in meta.columns:
        return meta
    if "week_start" not in meta.columns:
        raise KeyError("week_start_ts missing from meta and cannot derive from week_start.")
    meta = meta.copy()
    meta["week_start_ts"] = pd.to_datetime(meta["week_start"]).astype("int64") // 10**9
    return meta


def _write_dual_csv(df: pd.DataFrame, run_path: Path, latest_path: Path) -> None:
    run_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_path, index=False)
    if run_path != latest_path:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(latest_path, index=False)


def _copy_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _build_data_manifest(
    df: pd.DataFrame,
    meta_train: pd.DataFrame,
    meta_valid: pd.DataFrame,
    meta_test: pd.DataFrame,
    weekly_features_path: Path,
) -> dict:
    manifest = {
        "weekly_rows": int(len(df)),
        "unique_drugs": int(df["DrugId"].nunique()) if "DrugId" in df.columns else None,
        "unique_weeks": int(df["week_start"].nunique()) if "week_start" in df.columns else None,
        "week_start_range": _iso_range(df.get("week_start")),
        "train_rows": int(len(meta_train)),
        "valid_rows": int(len(meta_valid)),
        "test_rows": int(len(meta_test)),
        "train_week_range": _iso_range(meta_train.get("week_start")),
        "valid_week_range": _iso_range(meta_valid.get("week_start")),
        "test_week_range": _iso_range(meta_test.get("week_start")),
    }
    if weekly_features_path.exists():
        manifest["weekly_features_hash_md5"] = compute_file_hash(weekly_features_path)
    return manifest


def _split_weeks(
    meta: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weeks = np.sort(meta["week_start_ts"].unique())
    if len(weeks) < 3:
        raise ValueError("Need at least 3 unique weeks for train/valid/test split.")
    train_end = max(1, int(len(weeks) * train_frac))
    valid_end = max(train_end + 1, int(len(weeks) * (train_frac + valid_frac)))
    if valid_end >= len(weeks):
        valid_end = len(weeks) - 1
        train_end = max(1, valid_end - 1)
    train_weeks = weeks[:train_end]
    valid_weeks = weeks[train_end:valid_end]
    test_weeks = weeks[valid_end:]
    if len(test_weeks) == 0:
        raise ValueError("Test split is empty; adjust split fractions.")
    return train_weeks, valid_weeks, test_weeks


def _three_year_split(
    meta: pd.DataFrame,
    min_valid_weeks: int = 4,
    min_test_weeks: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Year-based split: years 1-2 train, year 3 H1 valid, year 3 H2 test."""
    weeks_df = meta[["week_start_ts", "week_start"]].drop_duplicates().sort_values("week_start_ts")
    if weeks_df.empty:
        raise ValueError("No weekly information available for splitting.")

    week_dates = pd.to_datetime(weeks_df["week_start"])
    base_year = int(week_dates.min().year)
    weeks_df["rel_year"] = week_dates.dt.year - base_year + 1
    weeks_df["month"] = week_dates.dt.month

    train_weeks = weeks_df.loc[weeks_df["rel_year"].isin([1, 2]), "week_start_ts"].to_numpy()
    valid_weeks = weeks_df.loc[(weeks_df["rel_year"] == 3) & (weeks_df["month"] <= 6), "week_start_ts"].to_numpy()
    test_weeks = weeks_df.loc[(weeks_df["rel_year"] == 3) & (weeks_df["month"] >= 7), "week_start_ts"].to_numpy()

    if len(valid_weeks) < min_valid_weeks or len(test_weeks) < min_test_weeks or len(train_weeks) == 0:
        return _split_weeks(meta, train_frac=0.7, valid_frac=0.15)

    return train_weeks, valid_weeks, test_weeks


def compute_sample_weights(
    meta: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    half_life_weeks: int = 26,
    min_drug_weight: float = 0.6,
    max_drug_weight: float = 2.0,
) -> np.ndarray:
    """Combine time-decay and volume weights; downweight censored rows if present."""
    if len(meta) == 0:
        return np.array([])

    # Time decay (recent weeks get higher weight)
    weeks_sorted = np.sort(meta["week_start_ts"].unique())
    week_rank = {wk: i for i, wk in enumerate(weeks_sorted)}
    max_rank = max(week_rank.values()) if week_rank else 0
    decay = np.log(2) / max(half_life_weeks, 1)
    time_weight = meta["week_start_ts"].map(
        lambda wk: float(np.exp(decay * (week_rank[wk] - max_rank)))
    ).to_numpy()

    # Volume weight based on train history only
    train_meta = meta.loc[train_mask].reset_index(drop=True)
    train_y = y.loc[train_mask].reset_index(drop=True)
    drug_totals = train_y.groupby(train_meta["DrugId"]).sum()
    median_total = float(drug_totals.median()) if not drug_totals.empty else 1.0
    drug_scale = (drug_totals / max(median_total, 1e-6)) ** 0.5
    drug_scale = drug_scale.clip(lower=min_drug_weight, upper=max_drug_weight)
    drug_weight = meta["DrugId"].map(drug_scale).fillna(1.0).to_numpy()

    weights = time_weight * drug_weight

    # Optional censored flag
    for col in ["censored_flag", "stockout_flag", "is_censored"]:
        if col in meta.columns:
            weights = np.where(meta[col].fillna(0).to_numpy() > 0, weights * 0.5, weights)
            break

    return weights


def _select_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
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


def audit_feature_leakage(
    df: pd.DataFrame,
    output_path: Path | None = None,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Recompute rolling features from lagged UPW and report mismatches."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    required = {"DrugId", "week_start", "UPW"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Leakage audit requires columns: {sorted(missing)}")

    work = df.copy().sort_values(["DrugId", "week_start"]).reset_index(drop=True)
    work["week_of_year"] = pd.to_datetime(work["week_start"]).dt.isocalendar().week.astype(int)
    rows = []

    def _recompute(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("week_start").copy()
        lag_series = g["UPW"].shift(1)
        g["UPW_rollmean_4_chk"] = lag_series.rolling(window=4, min_periods=4).mean()
        g["UPW_rollmean_8_chk"] = lag_series.rolling(window=8, min_periods=8).mean()
        g["UPW_rollmean_12_chk"] = lag_series.rolling(window=12, min_periods=12).mean()
        g["UPW_rollmean_52_chk"] = lag_series.rolling(window=52, min_periods=26).mean()
        g["UPW_rollmax_8_chk"] = lag_series.rolling(window=8, min_periods=8).max()
        g["UPW_rollstd_8_chk"] = lag_series.rolling(window=8, min_periods=8).std()
        g["diff_rate_chk"] = lag_series.pct_change()
        roll_mean_12 = lag_series.rolling(window=12, min_periods=12).mean()
        roll_std_12 = lag_series.rolling(window=12, min_periods=12).std()
        g["Z_score_chk"] = (lag_series - roll_mean_12) / (roll_std_12 + epsilon)
        g["weeks_since_peak_13_chk"] = (
            lag_series.rolling(window=13, min_periods=4).apply(
                lambda s: len(s) - 1 - int(np.argmax(s.values))
            )
        )
        g["spike_score_8_chk"] = (lag_series - g["UPW_rollmean_8_chk"]) / (g["UPW_rollstd_8_chk"] + epsilon)
        g["diff_rate_chk"] = g["diff_rate_chk"].clip(lower=-3, upper=3)
        g["Z_score_chk"] = g["Z_score_chk"].clip(lower=-3, upper=3)
        regime_raw = ((g["week_of_year"] <= 8) | (g["week_of_year"] >= 45)).astype(float)
        regime_lag = regime_raw.shift(1)
        g["regime_prob_8w_chk"] = regime_lag.rolling(window=8, min_periods=4).mean()
        g["spike_flag_chk"] = (
            (lag_series > 1.5 * (g["UPW_rollmean_12_chk"] + epsilon)) | (g["diff_rate_chk"] > 1.0)
        ).astype(int)
        return g

    audit = work.groupby("DrugId", group_keys=False).apply(_recompute, include_groups=False)
    feature_pairs = {
        "UPW_rollmean_4": "UPW_rollmean_4_chk",
        "UPW_rollmean_8": "UPW_rollmean_8_chk",
        "UPW_rollmean_12": "UPW_rollmean_12_chk",
        "UPW_rollmean_52": "UPW_rollmean_52_chk",
        "UPW_rollmax_8": "UPW_rollmax_8_chk",
        "UPW_rollstd_8": "UPW_rollstd_8_chk",
        "diff_rate": "diff_rate_chk",
        "Z_score": "Z_score_chk",
        "weeks_since_peak_13": "weeks_since_peak_13_chk",
        "spike_score_8": "spike_score_8_chk",
    }
    for col, chk in feature_pairs.items():
        if col not in audit.columns or chk not in audit.columns:
            continue
        lhs = audit[col].round(2)
        rhs = audit[chk].round(2)
        diff = (lhs - rhs).abs()
        mismatch = diff > 1e-2
        rows.append(
            {
                "feature": col,
                "max_abs_diff": float(diff.max(skipna=True)) if not diff.dropna().empty else 0.0,
                "mismatch_count": int(mismatch.sum()),
            }
        )
    report = pd.DataFrame(rows).sort_values(["mismatch_count", "max_abs_diff"], ascending=False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
    return report


def _baseline_predictions(
    y: pd.Series,
    meta: pd.DataFrame,
    ma_window: int = 4,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "DrugId": meta["DrugId"],
            "week_start_ts": meta["week_start_ts"],
            "y": y,
        },
        index=meta.index,
    )
    ordered = data.sort_values(["DrugId", "week_start_ts"]).reset_index()
    ordered["pred_naive"] = ordered.groupby("DrugId")["y"].shift(1)
    ordered["pred_moving_avg"] = (
        ordered.groupby("DrugId")["y"]
        .apply(lambda s: s.shift(1).rolling(window=ma_window, min_periods=ma_window).mean())
        .reset_index(level=0, drop=True)
    )
    ordered["pred_seasonal_naive"] = ordered.groupby("DrugId")["y"].shift(seasonal_period)
    preds = ordered.set_index("index")[
        ["pred_naive", "pred_moving_avg", "pred_seasonal_naive"]
    ].reindex(meta.index)
    return preds


def _evaluate_with_mask(
    y: pd.Series,
    preds: pd.Series,
    mask: pd.Series | np.ndarray,
    label: str,
) -> dict[str, float | int | None]:
    valid_mask = pd.Series(mask, index=y.index) & preds.notna()
    count = int(valid_mask.sum())
    if count == 0:
        return {
            "label": label,
            "n_obs": 0,
            "rmse": None,
            "mae": None,
            "median_ae": None,
            "r2": None,
            "bias": None,
            "wape": None,
            "smape": None,
        }
    metrics = evaluate_predictions(y[valid_mask], preds[valid_mask], label=label)
    metrics["n_obs"] = count
    return metrics


def prepare_features(df: pd.DataFrame):
    """Prepare lag-safe features for XGBoost; return X (unencoded), y, and meta."""
    data = df.copy().sort_values(["DrugId", "week_start"]).reset_index(drop=True)
    # Convert date to numeric timestamp (seconds since epoch)
    data["week_start_ts"] = pd.to_datetime(data["week_start"]).astype("int64") // 10**9

    week_num = data["week_of_year"]

    # Ensure lagged target exists (built upstream, but recompute defensively)
    if "UPW_lag1" not in data.columns:
        data["UPW_lag1"] = data.groupby("DrugId")["UPW"].shift(1)

    # Ensure safe rolling stats exist (lagged only)
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

    # Spike score for policy use (not model feature)
    data["spike_score_8"] = (lag_series - data["UPW_rollmean_8"]) / (data["UPW_rollstd_8"] + 1e-6)

    # Drop rows without sufficient lag history
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

    # Target and features
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
        c for c in X_full.columns if X_full[c].dtype == "object" or str(X_full[c].dtype).startswith("category")
    ]
    keep_cols = [c for c in safe_features if c in X_full.columns] + categorical_cols
    X = X_full[keep_cols].copy()

    return X, y, meta


def encode_categoricals(
    X_train: pd.DataFrame, X_valid: pd.DataFrame
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


def apply_label_encoders(X: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
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


def train_xgb_model(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    params: dict | None = None,
    random_seed: int | None = None,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
):
    """Champion training: raw target + weights + safe features (no two-stage, no shock in-model)."""
    meta = _ensure_week_start_ts(meta)
    order = np.argsort(meta["week_start_ts"].values)
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)
    meta_ordered = meta.iloc[order].reset_index(drop=True)

    train_weeks, valid_weeks, test_weeks = _three_year_split(
        meta_ordered, min_valid_weeks=4, min_test_weeks=4
    )
    train_mask = meta_ordered["week_start_ts"].isin(train_weeks).to_numpy()
    valid_mask = meta_ordered["week_start_ts"].isin(valid_weeks).to_numpy()
    test_mask = meta_ordered["week_start_ts"].isin(test_weeks).to_numpy()

    weights_ordered = compute_sample_weights(meta_ordered, y_ordered, train_mask=train_mask)

    X_train_raw = X_ordered.loc[train_mask].reset_index(drop=True)
    X_valid_raw = X_ordered.loc[valid_mask].reset_index(drop=True)
    X_test_raw = X_ordered.loc[test_mask].reset_index(drop=True)
    y_train = y_ordered.loc[train_mask].reset_index(drop=True)
    y_valid = y_ordered.loc[valid_mask].reset_index(drop=True)
    y_test = y_ordered.loc[test_mask].reset_index(drop=True)
    meta_train = meta_ordered.loc[train_mask].reset_index(drop=True)
    meta_valid = meta_ordered.loc[valid_mask].reset_index(drop=True)
    meta_test = meta_ordered.loc[test_mask].reset_index(drop=True)
    w_train = weights_ordered[train_mask]
    w_valid = weights_ordered[valid_mask]
    w_test = weights_ordered[test_mask]

    X_train, X_valid, encoders = encode_categoricals(X_train_raw, X_valid_raw)
    X_test = apply_label_encoders(X_test_raw, encoders)

    train_dm = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    valid_dm = xgb.DMatrix(X_valid, label=y_valid, weight=w_valid)
    test_dm = xgb.DMatrix(X_test, label=y_test, weight=w_test)

    active_params = dict(params) if params is not None else {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 8.0,
        "gamma": 0.4,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
    }
    if random_seed is not None:
        active_params.setdefault("seed", int(random_seed))

    evals = [(train_dm, "train"), (valid_dm, "valid")]
    model = xgb.train(
        params=active_params,
        dtrain=train_dm,
        num_boost_round=250,
        evals=evals,
        early_stopping_rounds=35,
        verbose_eval=False,
    )

    preds_train = np.clip(model.predict(train_dm), 0, None)
    preds_valid = np.clip(model.predict(valid_dm), 0, None)
    preds_test = np.clip(model.predict(test_dm), 0, None)

    # Top feature importances
    importance = model.get_score(importance_type="gain")
    top_importance = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop features (gain):")
    for name, score in top_importance:
        print(f"{name}: {score:.4f}")

    return model, preds_train, preds_valid, preds_test, (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        meta_train,
        meta_valid,
        meta_test,
        X_valid_raw,
        X_test_raw,
        encoders,
    )


def time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    meta: pd.DataFrame | None = None,
    n_splits: int = 3,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 40,
) -> list[dict[str, float]]:
    """Time-series CV for champion model (raw target, safe features)."""
    if meta is None:
        if "week_start_ts" not in X.columns or "DrugId" not in X.columns:
            raise KeyError("time_series_cv requires week_start_ts and DrugId in X or a meta DataFrame.")
        order = np.argsort(X["week_start_ts"].values)
        X_ordered = X.iloc[order].reset_index(drop=True)
        y_ordered = y.iloc[order].reset_index(drop=True)
        meta_ordered = X_ordered[["week_start_ts", "DrugId"]].copy()
    else:
        meta = _ensure_week_start_ts(meta)
        order = np.argsort(meta["week_start_ts"].values)
        X_ordered = X.iloc[order].reset_index(drop=True)
        y_ordered = y.iloc[order].reset_index(drop=True)
        meta_ordered = meta.iloc[order].reset_index(drop=True)
        if "DrugId" not in meta_ordered.columns:
            raise KeyError("time_series_cv requires DrugId in meta.")
    week_vals = meta_ordered["week_start_ts"].to_numpy()
    weeks = np.sort(np.unique(week_vals))
    if len(weeks) <= n_splits:
        raise ValueError("Not enough unique weeks for time-series CV.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_week_idx, valid_week_idx) in enumerate(tscv.split(weeks), start=1):
        train_weeks = weeks[train_week_idx]
        valid_weeks = weeks[valid_week_idx]
        train_mask = np.isin(week_vals, train_weeks)
        valid_mask = np.isin(week_vals, valid_weeks)

        X_train_raw, X_valid_raw = X_ordered.loc[train_mask], X_ordered.loc[valid_mask]
        y_train_raw, y_valid_raw = y_ordered.loc[train_mask], y_ordered.loc[valid_mask]
        w_all = compute_sample_weights(meta_ordered, y_ordered, train_mask=train_mask)
        w_train, w_valid = w_all[train_mask], w_all[valid_mask]

        X_train_enc, X_valid_enc, _ = encode_categoricals(X_train_raw, X_valid_raw)

        train_dm = xgb.DMatrix(X_train_enc, label=y_train_raw, weight=w_train)
        valid_dm = xgb.DMatrix(X_valid_enc, label=y_valid_raw, weight=w_valid)
        evals = [(train_dm, "train"), (valid_dm, "valid")]

        booster = xgb.train(
            params=params,
            dtrain=train_dm,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        preds = np.clip(booster.predict(valid_dm), 0, None)

        metrics = evaluate_predictions(y_valid_raw, preds, label=f"fold_{fold}")
        fold_metrics.append(metrics)
        print(
            f"Fold {fold}: RMSE={metrics['rmse']:.3f} | MAE={metrics['mae']:.3f} | "
            f"R2={metrics['r2']:.3f} | Bias={metrics['bias']:.3f}"
        )

    return fold_metrics


def monitor_drift(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_path: str = "data/processed/upw_fold_drift.csv",
) -> pd.DataFrame:
    """Basic drift check: compare means/stds of numeric features between train and valid."""
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    rows = []
    for col in num_cols:
        tr_mean, va_mean = train_df[col].mean(), valid_df[col].mean()
        tr_std, va_std = train_df[col].std(), valid_df[col].std()
        mean_diff = va_mean - tr_mean
        mean_pct = mean_diff / (abs(tr_mean) + 1e-6)
        std_diff = va_std - tr_std
        rows.append(
            {
                "feature": col,
                "train_mean": tr_mean,
                "valid_mean": va_mean,
                "mean_diff": mean_diff,
                "mean_pct": mean_pct,
                "train_std": tr_std,
                "valid_std": va_std,
                "std_diff": std_diff,
            }
        )
    drift_df = pd.DataFrame(rows).sort_values(by="mean_pct", key=np.abs, ascending=False)
    drift_df.to_csv(output_path, index=False)
    top5 = drift_df.head(5)
    print("\n=== Fold drift check (top 5 by mean % diff) ===")
    for _, row in top5.iterrows():
        print(
            f"{row['feature']}: train_mean={row['train_mean']:.3f}, valid_mean={row['valid_mean']:.3f}, "
            f"mean_pct_diff={row['mean_pct']:.2%}"
        )
    alerts = drift_df[drift_df["mean_pct"].abs() > 0.2]
    if not alerts.empty:
        print("\n[ALERT] Potential drift on feature means (>20% diff):")
        for _, row in alerts.iterrows():
            print(f" - {row['feature']}: mean_pct_diff={row['mean_pct']:.2%}")
    return drift_df


def apply_regime_calibration(
    preds: np.ndarray,
    meta_valid: pd.DataFrame,
    y_train: pd.Series,
    preds_train: np.ndarray,
    meta_train: pd.DataFrame,
    regime_col: str | None = None,
    recent_window: int = 8,
) -> np.ndarray:
    """Bias-correct predictions; optionally conditioned on a soft regime column."""
    residuals_train = preds_train - y_train
    recent_bias = float(residuals_train.tail(recent_window).mean()) if len(residuals_train) else 0.0

    if regime_col and regime_col in meta_train.columns and regime_col in meta_valid.columns:
        # Bin soft regime into deciles to avoid over-fragmentation
        def _bin(series: pd.Series) -> pd.Series:
            return (series.fillna(0) * 10).round().clip(0, 10)

        train_bins = _bin(meta_train[regime_col])
        valid_bins = _bin(meta_valid[regime_col])
        bias_map = {
            bin_val: float(residuals_train[train_bins == bin_val].mean())
            for bin_val in np.unique(train_bins)
        }
        adj = np.array([bias_map.get(b, recent_bias) for b in valid_bins])
        return preds + adj

    return preds + recent_bias


def apply_shock_guard(
    preds: np.ndarray,
    base_pred: np.ndarray,
    X_raw: pd.DataFrame,
    diff_threshold: float = 2.5,
    z_threshold: float = 2.5,
    peak_scale: float = 1.1,
) -> np.ndarray:
    """Override predictions during detected shocks using baseline and recent peak."""
    preds_adj = np.array(preds, copy=True)
    base_arr = np.asarray(base_pred)
    diff = X_raw.get("diff_rate")
    zscore = X_raw.get("Z_score")
    spike_score = X_raw.get("spike_score_8")
    if diff is None and zscore is None and spike_score is None:
        return np.clip(preds_adj, 0, None)
    mask = np.zeros(len(preds_adj), dtype=bool)
    if diff is not None:
        mask |= diff.abs().to_numpy() > diff_threshold
    if zscore is not None:
        mask |= zscore.abs().to_numpy() > z_threshold
    if spike_score is not None:
        mask |= spike_score.abs().to_numpy() > z_threshold
    if not mask.any():
        return np.clip(preds_adj, 0, None)
    peak = X_raw.get("UPW_rollmax_8", pd.Series(0, index=X_raw.index)).fillna(0).to_numpy()
    candidate = np.maximum(base_arr, peak_scale * peak)
    preds_adj[mask] = np.maximum(preds_adj[mask], candidate[mask])
    return np.clip(preds_adj, 0, None)


def summarize_walkforward_fold1(
    wf_df: pd.DataFrame,
    meta: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Summarize the first walk-forward week and its drug composition."""
    if wf_df.empty:
        return pd.DataFrame()
    first_week = wf_df["week_start_ts"].min()
    fold1 = wf_df[wf_df["week_start_ts"] == first_week].copy()
    first_seen = meta.groupby("DrugId")["week_start_ts"].min()
    fold1["is_new_drug"] = fold1["DrugId"].map(first_seen) == first_week
    fold1 = fold1.sort_values("abs_err", ascending=False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fold1.to_csv(output_path, index=False)
    print(
        "\n=== Walk-forward Fold1 summary ===\n"
        f"Week_ts={int(first_week)} | Drugs={fold1['DrugId'].nunique()} | "
        f"New drugs={int(fold1['is_new_drug'].sum())} | Top abs_err={fold1['abs_err'].max():.3f}"
    )
    return fold1


def monitor_rolling_rmse(y_true: pd.Series, y_pred: np.ndarray, window: int = 8) -> None:
    """Simple rolling RMSE monitor; alerts if last exceeds 2x historical median."""
    residuals = (y_true - y_pred) ** 2
    rmse_roll = np.sqrt(residuals.rolling(window=window, min_periods=window).mean())
    if rmse_roll.dropna().empty:
        return
    median_hist = rmse_roll.median()
    last_rmse = rmse_roll.dropna().iloc[-1]
    print(f"\nRolling RMSE (window={window}) last={last_rmse:.3f}, median={median_hist:.3f}")
    if last_rmse > 2 * median_hist:
        print("[ALERT] Rolling RMSE exceeds 2x historical median. Possible performance drift.")


def walk_forward_forecast(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 40,
    min_train_weeks: int = 30,
    error_window: int = 8,
    output_path: str = "data/processed/upw_walkforward_predictions.csv",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rolling-origin training for champion model (raw target, safe features)."""
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": ["rmse", "mae"],
            "eta": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 8.0,
            "gamma": 0.4,
            "reg_lambda": 2.0,
            "reg_alpha": 0.1,
        }

    meta = _ensure_week_start_ts(meta)
    order = np.argsort(meta["week_start_ts"].values)
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)
    meta_ordered = meta.iloc[order].reset_index(drop=True)

    unique_weeks = np.sort(meta_ordered["week_start_ts"].unique())
    start_idx = min_train_weeks
    if len(unique_weeks) <= start_idx:
        raise ValueError("Not enough weekly history to run walk-forward backtest.")

    rows: list[pd.DataFrame] = []
    for wk_ts in unique_weeks[start_idx:]:
        train_mask = meta_ordered["week_start_ts"] < wk_ts
        valid_mask = meta_ordered["week_start_ts"] == wk_ts
        if valid_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train_all = X_ordered.loc[train_mask].reset_index(drop=True)
        if len(X_train_all) < 2:
            continue
        y_train_all = y_ordered.loc[train_mask].reset_index(drop=True)
        weights_all = compute_sample_weights(meta_ordered, y_ordered, train_mask=train_mask)
        w_train_all = weights_all[train_mask]

        inner_split = max(int(len(X_train_all) * 0.85), len(X_train_all) - 150)
        inner_split = min(max(inner_split, 1), len(X_train_all) - 1)
        X_inner_train_raw = X_train_all.iloc[:inner_split]
        X_inner_valid_raw = X_train_all.iloc[inner_split:]
        y_inner_train = y_train_all.iloc[:inner_split]
        y_inner_valid = y_train_all.iloc[inner_split:]
        w_inner_train = w_train_all[:inner_split]
        w_inner_valid = w_train_all[inner_split:]

        X_inner_train, X_inner_valid, encoders = encode_categoricals(
            X_inner_train_raw, X_inner_valid_raw
        )
        train_dm = xgb.DMatrix(X_inner_train, label=y_inner_train, weight=w_inner_train)
        valid_dm = xgb.DMatrix(X_inner_valid, label=y_inner_valid, weight=w_inner_valid)
        booster = xgb.train(
            params=params,
            dtrain=train_dm,
            num_boost_round=num_boost_round,
            evals=[(train_dm, "train"), (valid_dm, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        X_valid_raw = X_ordered.loc[valid_mask].reset_index(drop=True)
        y_valid = y_ordered.loc[valid_mask].reset_index(drop=True)
        meta_valid = meta_ordered.loc[valid_mask].reset_index(drop=True)
        X_valid_enc = apply_label_encoders(X_valid_raw, encoders)
        valid_dm = xgb.DMatrix(X_valid_enc)

        preds_valid = np.clip(booster.predict(valid_dm), 0, None)

        week_df = meta_valid.copy()
        if "week_start_ts" not in week_df.columns and "week_start_ts" in X_valid_raw.columns:
            week_df["week_start_ts"] = X_valid_raw["week_start_ts"].values
        feature_cols = [
            "UPW_lag1",
            "UPW_rollmean_4",
            "UPW_rollmean_8",
            "UPW_rollmean_12",
            "UPW_rollmax_8",
            "UPW_rollstd_8",
            "weeks_since_peak_13",
            "spike_score_8",
        ]
        keep_cols = [c for c in feature_cols if c in X_valid_raw.columns]
        if keep_cols:
            week_df = pd.concat(
                [week_df.reset_index(drop=True), X_valid_raw[keep_cols].reset_index(drop=True)],
                axis=1,
            )
        week_df["y_true"] = y_valid.values
        week_df["y_pred"] = preds_valid
        week_df["y_pred_cal"] = preds_valid
        week_df["abs_err"] = np.abs(week_df["y_pred"] - week_df["y_true"])
        rows.append(week_df)

    if not rows:
        raise ValueError("No walk-forward predictions produced; check date coverage.")

    wf_df = pd.concat(rows, ignore_index=True).sort_values(["week_start_ts", "DrugId"]).reset_index(drop=True)
    eps = 1e-6
    min_periods = max(2, error_window // 2)
    wf_df["rolling_mae"] = (
        wf_df.groupby("DrugId")["abs_err"]
        .apply(lambda s: s.rolling(window=error_window, min_periods=min_periods).mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    wf_df["rolling_mean_y"] = (
        wf_df.groupby("DrugId")["y_true"]
        .apply(lambda s: s.rolling(window=error_window, min_periods=min_periods).mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    wf_df["error_rate_roll"] = wf_df["rolling_mae"] / (wf_df["rolling_mean_y"].abs() + eps)
    global_error_rate = float(wf_df["abs_err"].mean() / (wf_df["y_true"].abs().mean() + eps))
    global_error_rate = float(np.clip(global_error_rate, 0.0, 1.0))
    wf_df["error_rate"] = wf_df["error_rate_roll"].fillna(global_error_rate)
    wf_df["error_rate"] = wf_df["error_rate"].clip(lower=0.0, upper=1.0)

    pred_low, pred_mid, pred_high = build_error_band(
        wf_df["y_pred_cal"].to_numpy(), wf_df["error_rate"].to_numpy()
    )
    wf_df["y_pred_low"] = np.round(pred_low).astype(int)
    wf_df["y_pred_mid"] = np.round(pred_mid).astype(int)
    wf_df["y_pred_high"] = np.round(pred_high).astype(int)
    wf_df["y_pred"] = np.round(wf_df["y_pred"]).astype(int)
    low_value, mid_value, high_value = compute_decile_values(
        wf_df["y_pred_low"].to_numpy(),
        wf_df["y_pred_high"].to_numpy(),
    )
    wf_df["low_value"] = low_value
    wf_df["mid_value"] = mid_value
    wf_df["high_value"] = high_value

    wf_df.to_csv(output_path, index=False)
    wf_metrics = evaluate_predictions(wf_df["y_true"], wf_df["y_pred_cal"], label="walk_forward")
    print(
        f"\n=== Walk-forward backtest ===\n"
        f"RMSE={wf_metrics['rmse']:.3f} | MAE={wf_metrics['mae']:.3f} | "
        f"MedAE={wf_metrics['median_ae']:.3f} | R2={wf_metrics['r2']:.3f}"
    )
    print(
        f"Saved per-week predictions and rolling error bands to {output_path} "
        f"(window={error_window}, min_train_weeks={min_train_weeks})."
    )
    return wf_df, wf_metrics


def evaluate_predictions(y_true: pd.Series, preds: np.ndarray, label: str = "") -> dict[str, float]:
    """Compute a set of regression metrics."""
    eps = 1e-6
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    medae = median_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    bias = float(np.mean(preds - y_true))
    wape = float(np.sum(np.abs(y_true - preds)) / (np.sum(np.abs(y_true)) + eps))
    smape = float(
        np.mean(2 * np.abs(y_true - preds) / (np.abs(y_true) + np.abs(preds) + eps))
    )
    # Weighted WAPE: heavier on top-demand weeks (>=75th percentile)
    high_cut = np.quantile(np.abs(y_true), 0.75) if len(y_true) else 0
    weights = np.where(np.abs(y_true) >= high_cut, 2.0, 1.0)
    wwape = float(
        np.sum(weights * np.abs(y_true - preds)) / (np.sum(weights * np.abs(y_true)) + eps)
    )
    under_mask = preds < y_true
    under_rate = float(np.mean(under_mask)) if len(y_true) else 0.0
    under_share = float(
        np.sum((y_true - preds) * under_mask) / (np.sum(np.abs(y_true)) + eps)
    )
    metrics = {
        "label": label,
        "rmse": rmse,
        "mae": mae,
        "median_ae": medae,
        "r2": r2,
        "bias": bias,
        "wape": wape,
        "smape": smape,
        "wwape": wwape,
        "under_forecast_rate": under_rate,
        "under_forecast_share": under_share,
    }
    return metrics


def build_error_band(
    preds: np.ndarray, error_rate: float | np.ndarray | pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (low, mid, high) predictions using a symmetric percentage band."""
    rates = np.asarray(error_rate, dtype=float)
    if np.any(rates < 0):
        raise ValueError("error_rate must be non-negative")
    rates = np.clip(rates, 0.0, 1.0)
    mid = preds
    delta = np.abs(preds) * rates
    low = np.clip(mid - delta, 0, None)
    high = mid + delta
    return low, mid, high


def compute_decile_values(
    pred_low: np.ndarray | pd.Series, pred_high: np.ndarray | pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the rounded [low, high] range into deciles and return 3/4/3 min-max ranges."""
    low_round = np.round(np.asarray(pred_low, dtype=float))
    high_round = np.round(np.asarray(pred_high, dtype=float))
    low_base = np.minimum(low_round, high_round)
    high_base = np.maximum(low_round, high_round)
    step = (high_base - low_base) / 10.0
    low_min = np.round(low_base).astype(int)
    low_max = np.round(low_base + 3 * step).astype(int)
    mid_min = np.round(low_base + 3 * step).astype(int)
    mid_max = np.round(low_base + 7 * step).astype(int)
    high_min = np.round(low_base + 7 * step).astype(int)
    high_max = np.round(high_base).astype(int)
    low_range = np.array([f"{a}-{b}" for a, b in zip(low_min, low_max)], dtype=object)
    mid_range = np.array([f"{a}-{b}" for a, b in zip(mid_min, mid_max)], dtype=object)
    high_range = np.array([f"{a}-{b}" for a, b in zip(high_min, high_max)], dtype=object)
    return low_range, mid_range, high_range


def calibrate_per_drug_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drug_ids: np.ndarray,
    clip_range: tuple[float, float] = (0.5, 1.5),
) -> dict:
    """Fit per-drug multiplicative scale on validation data."""
    scales: dict = {}
    for drug in np.unique(drug_ids):
        mask = drug_ids == drug
        denom = float(np.sum(y_pred[mask] ** 2))
        if denom <= 0:
            scales[drug] = 1.0
            continue
        a = float(np.sum(y_true[mask] * y_pred[mask]) / denom)
        a = float(np.clip(a, clip_range[0], clip_range[1]))
        scales[drug] = a
    return scales


def apply_per_drug_scale(
    y_pred: np.ndarray,
    drug_ids: np.ndarray,
    scales: dict,
) -> np.ndarray:
    scaled = np.array(y_pred, copy=True)
    for i, drug in enumerate(drug_ids):
        scaled[i] = scaled[i] * scales.get(drug, 1.0)
    return np.clip(scaled, 0, None)


def compute_residual_quantiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drug_ids: np.ndarray,
    quantiles: tuple[float, float] = (0.8, 0.9),
) -> dict:
    """Compute per-drug residual quantiles for inventory-style bands."""
    q_map: dict = {}
    for drug in np.unique(drug_ids):
        mask = drug_ids == drug
        residuals = y_true[mask] - y_pred[mask]
        if residuals.size == 0:
            q_map[drug] = (0.0, 0.0)
            continue
        q_vals = np.quantile(residuals, quantiles)
        q_map[drug] = tuple(float(v) for v in q_vals)
    return q_map


def apply_residual_quantiles(
    y_pred: np.ndarray,
    drug_ids: np.ndarray,
    q_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    q80 = np.array(y_pred, copy=True)
    q90 = np.array(y_pred, copy=True)
    for i, drug in enumerate(drug_ids):
        q_vals = q_map.get(drug, (0.0, 0.0))
        q80[i] = y_pred[i] + q_vals[0]
        q90[i] = y_pred[i] + q_vals[1]
    return np.clip(q80, 0, None), np.clip(q90, 0, None)


def apply_spike_policy(
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    y_true: np.ndarray | None = None,
    k_values: Iterable[float] = (1.0, 1.1, 1.2),
    min_z: float = 2.0,
) -> np.ndarray:
    """Post-process predictions for spike weeks using per-drug thresholds."""
    if "spike_score_8" not in meta.columns or "UPW_rollmax_8" not in meta.columns:
        return y_pred

    drug_ids = meta["DrugId"].to_numpy()
    spike_score = meta["spike_score_8"].to_numpy()
    rollmax_8 = meta["UPW_rollmax_8"].to_numpy()

    # Per-drug threshold from train/valid distribution
    z_thr = {}
    for drug in np.unique(drug_ids):
        vals = spike_score[drug_ids == drug]
        if len(vals) == 0:
            z_thr[drug] = min_z
        else:
            z_thr[drug] = max(min_z, float(np.quantile(vals, 0.9)))

    # Global k selection on spikes
    best_k = list(k_values)[0]
    if y_true is not None:
        best_wape = float("inf")
        for k in k_values:
            adjusted = np.array(y_pred, copy=True)
            for i, drug in enumerate(drug_ids):
                if spike_score[i] > z_thr.get(drug, min_z):
                    adjusted[i] = max(adjusted[i], rollmax_8[i] * k)
            wape = float(np.sum(np.abs(y_true - adjusted)) / (np.sum(np.abs(y_true)) + 1e-6))
            if wape < best_wape:
                best_wape = wape
                best_k = k

    adjusted = np.array(y_pred, copy=True)
    for i, drug in enumerate(drug_ids):
        if spike_score[i] > z_thr.get(drug, min_z):
            adjusted[i] = max(adjusted[i], rollmax_8[i] * best_k)
    return np.clip(adjusted, 0, None)


def shuffle_target_test(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    num_boost_round: int = 120,
    early_stopping_rounds: int = 20,
    random_seed: int | None = None,
) -> dict[str, float]:
    """Leakage check: shuffle targets and retrain; metrics should be near-zero signal."""
    meta = _ensure_week_start_ts(meta)
    order = np.argsort(meta["week_start_ts"].values)
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)

    y_shuffled = y_ordered.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(X_ordered) * 0.8)
    X_train_raw, X_test_raw = X_ordered.iloc[:split_idx], X_ordered.iloc[split_idx:]
    y_train_s, y_test_s = y_shuffled.iloc[:split_idx], y_shuffled.iloc[split_idx:]
    w = np.linspace(0.7, 1.3, len(X_ordered))
    w_train, w_test = w[:split_idx], w[split_idx:]

    X_train_enc, X_test_enc, _ = encode_categoricals(X_train_raw, X_test_raw)
    train_dm = xgb.DMatrix(X_train_enc, label=y_train_s, weight=w_train)
    test_dm = xgb.DMatrix(X_test_enc, label=y_test_s, weight=w_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 6.0,
        "gamma": 0.4,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
    }
    if random_seed is not None:
        params["seed"] = int(random_seed)

    booster = xgb.train(
        params=params,
        dtrain=train_dm,
        num_boost_round=num_boost_round,
        evals=[(train_dm, "train"), (test_dm, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    preds = booster.predict(test_dm)
    return evaluate_predictions(y_test_s, preds, label="shuffle_valid")


def print_evaluation(
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
    test_metrics: dict[str, float] | None = None,
) -> None:
    """Pretty-print metrics for train/validation/test splits."""
    def _fmt(m: dict[str, float]) -> str:
        return (
            f"RMSE={m['rmse']:.3f} | MAE={m['mae']:.3f} | MedAE={m['median_ae']:.3f} | "
            f"R2={m['r2']:.3f} | WAPE={m['wape']:.3f} | WWAPE={m.get('wwape', 0):.3f} | "
            f"sMAPE={m['smape']:.3f} | Bias={m['bias']:.3f} | "
            f"UnderRate={m.get('under_forecast_rate', 0):.2f} | "
            f"UnderShare={m.get('under_forecast_share', 0):.3f}"
        )

    print("\n=== Evaluation ===")
    print(f"Train: {_fmt(train_metrics)}")
    print(f"Valid: {_fmt(valid_metrics)}")
    if test_metrics is not None:
        print(f"Test: {_fmt(test_metrics)}")


def main(
    settings: Settings | None = None,
    data: pd.DataFrame | None = None,
    run_tag: str | None = None,
    run_dir: Path | None = None,
    run_walk_forward: bool | None = None,
) -> dict[str, float]:
    active_settings = settings or get_settings()
    set_global_seed(active_settings.random_seed)
    active_run_walk_forward = (
        run_walk_forward
        if run_walk_forward is not None
        else active_settings.run_walk_forward
    )

    artifacts_dir = active_settings.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = active_settings.data_path() / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    active_run_dir = run_dir or create_run_dir(artifacts_dir, run_tag=run_tag)
    run_metadata = {
        "run_id": active_run_dir.name,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "seed": active_settings.random_seed,
    }
    write_json(active_run_dir / "run.json", run_metadata)
    write_json(active_run_dir / "config.json", _safe_settings_snapshot(active_settings))

    df = data if data is not None else build_dataset(settings=active_settings)
    weekly_features_path = processed_dir / "weekly_features.csv"

    X, y, meta = prepare_features(df)
    diagnostics_dir = active_run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    audit_df = pd.concat([meta, X, y.rename("UPW")], axis=1)
    audit_report = audit_feature_leakage(
        audit_df, output_path=diagnostics_dir / "feature_leakage_audit.csv"
    )
    if not audit_report.empty and audit_report["mismatch_count"].sum() > 0:
        print("\n[WARN] Leakage audit found mismatches; see diagnostics/feature_leakage_audit.csv")
    model_params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "min_child_weight": 8.0,
        "gamma": 0.4,
        "reg_lambda": 1.5,
        "reg_alpha": 0.1,
        "seed": int(active_settings.random_seed),
    }
    model, preds_train, preds_valid, preds_test, splits = train_xgb_model(
        X,
        y,
        meta,
        params=model_params,
        random_seed=active_settings.random_seed,
        train_frac=0.7,
        valid_frac=0.15,
    )
    (
        X_train_enc,
        X_valid_enc,
        X_test_enc,
        y_train,
        y_valid,
        y_test,
        meta_train,
        meta_valid,
        meta_test,
        X_valid_raw,
        X_test_raw,
        _encoders,
    ) = splits

    print(
        "\n=== Split summary ===\n"
        f"Train weeks: {_iso_range(meta_train['week_start'])} | Drugs={meta_train['DrugId'].nunique()}\n"
        f"Valid weeks: {_iso_range(meta_valid['week_start'])} | Drugs={meta_valid['DrugId'].nunique()}\n"
        f"Test weeks:  {_iso_range(meta_test['week_start'])} | Drugs={meta_test['DrugId'].nunique()}"
    )

    train_metrics = evaluate_predictions(y_train, preds_train, label="train")
    valid_metrics = evaluate_predictions(y_valid, preds_valid, label="valid")
    test_metrics = evaluate_predictions(y_test, preds_test, label="test")
    scales = calibrate_per_drug_scale(
        y_valid.to_numpy(),
        preds_valid,
        meta_valid["DrugId"].to_numpy(),
    )
    preds_valid_cal = apply_per_drug_scale(
        preds_valid, meta_valid["DrugId"].to_numpy(), scales
    )
    preds_test_cal = apply_per_drug_scale(
        preds_test, meta_test["DrugId"].to_numpy(), scales
    )
    if ENABLE_SPIKE_POLICY:
        preds_valid_cal = apply_spike_policy(
            preds_valid_cal, meta_valid, y_true=y_valid.to_numpy()
        )
        preds_test_cal = apply_spike_policy(preds_test_cal, meta_test, y_true=None)
    abs_err = np.abs(preds_valid_cal - y_valid.to_numpy())
    error_df = pd.DataFrame(
        {
            "DrugId": meta_valid["DrugId"].to_numpy(),
            "abs_err": abs_err,
            "y_true": y_valid.to_numpy(),
        }
    )
    per_drug = (
        error_df.groupby("DrugId", as_index=False)
        .agg(mae=("abs_err", "mean"), mean_y=("y_true", "mean"))
    )
    eps = 1e-6
    per_drug["error_rate"] = per_drug["mae"] / (per_drug["mean_y"].abs() + eps)
    per_drug["error_rate"] = per_drug["error_rate"].clip(lower=0.0, upper=1.0)
    global_error_rate = float(abs_err.mean() / (np.abs(y_valid).mean() + eps))
    global_error_rate = float(np.clip(global_error_rate, 0.0, 1.0))
    error_rate_series = meta_valid["DrugId"].map(
        per_drug.set_index("DrugId")["error_rate"]
    ).fillna(global_error_rate)
    error_rate_series = error_rate_series.clip(lower=0.0, upper=1.0)
    pred_low, pred_mid, pred_high = build_error_band(preds_valid_cal, error_rate_series.to_numpy())
    valid_cal_metrics = evaluate_predictions(y_valid, preds_valid_cal, label="valid_calibrated")
    test_cal_metrics = evaluate_predictions(y_test, preds_test_cal, label="test_calibrated")
    print_evaluation(train_metrics, valid_metrics, test_metrics)
    print("Valid metrics (per-drug scale):")
    print(
        f"RMSE={valid_cal_metrics['rmse']:.3f} | MAE={valid_cal_metrics['mae']:.3f} | "
        f"MedAE={valid_cal_metrics['median_ae']:.3f} | R2={valid_cal_metrics['r2']:.3f} | "
        f"WAPE={valid_cal_metrics['wape']:.3f} | WWAPE={valid_cal_metrics['wwape']:.3f} | "
        f"sMAPE={valid_cal_metrics['smape']:.3f} | Bias={valid_cal_metrics['bias']:.3f} | "
        f"UnderRate={valid_cal_metrics['under_forecast_rate']:.2f}"
    )
    print("Test metrics (per-drug scale):")
    print(
        f"RMSE={test_cal_metrics['rmse']:.3f} | MAE={test_cal_metrics['mae']:.3f} | "
        f"MedAE={test_cal_metrics['median_ae']:.3f} | R2={test_cal_metrics['r2']:.3f} | "
        f"WAPE={test_cal_metrics['wape']:.3f} | WWAPE={test_cal_metrics['wwape']:.3f} | "
        f"sMAPE={test_cal_metrics['smape']:.3f} | Bias={test_cal_metrics['bias']:.3f} | "
        f"UnderRate={test_cal_metrics['under_forecast_rate']:.2f}"
    )

    # Time-series cross-validation summary
    print("\n=== Time-series cross-validation (rolling splits) ===")
    cv_metrics = time_series_cv(
        X,
        y,
        params=model_params,
        meta=meta,
        n_splits=3,
        num_boost_round=300,
        early_stopping_rounds=40,
    )
    avg_rmse = np.mean([m["rmse"] for m in cv_metrics])
    avg_mae = np.mean([m["mae"] for m in cv_metrics])
    print(f"Average CV RMSE: {avg_rmse:.3f} | Average CV MAE: {avg_mae:.3f}")

    pred_dir = active_run_dir / "predictions"
    diagnostics_dir = active_run_dir / "diagnostics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    wf_metrics: dict[str, float] | None = None
    if active_run_walk_forward:
        run_wf_path = pred_dir / "upw_walkforward_predictions.csv"
        print("\n=== Walk-forward (rolling-origin) with rolling MAE-based bands ===")
        wf_df, wf_metrics = walk_forward_forecast(
            X,
            y,
            meta,
            params=model_params,
            num_boost_round=300,
            early_stopping_rounds=40,
            min_train_weeks=30,
            error_window=8,
            output_path=str(run_wf_path),
        )
        fold1_path = diagnostics_dir / "walkforward_fold1_drugs.csv"
        summarize_walkforward_fold1(wf_df, meta, output_path=fold1_path)
        _copy_if_exists(fold1_path, processed_dir / fold1_path.name)
        _copy_if_exists(run_wf_path, processed_dir / run_wf_path.name)
        print(
            f"Walk-forward: RMSE={wf_metrics['rmse']:.3f} | MAE={wf_metrics['mae']:.3f} | "
            f"Median error_rate={wf_df['error_rate'].median():.3f}"
        )
    else:
        print("\n[Skip] Walk-forward backtest disabled.")

    # Export validation preview with meta + features + predictions/residuals
    valid_preview = meta_valid.reset_index(drop=True).copy()
    features_preview = X_valid_raw.reset_index(drop=True).copy()
    valid_preview = pd.concat([valid_preview, features_preview], axis=1)
    valid_preview["y_true"] = y_valid.values
    valid_preview["y_pred"] = preds_valid
    valid_preview["y_pred_cal"] = preds_valid_cal
    q_map = compute_residual_quantiles(
        y_valid.to_numpy(),
        preds_valid_cal,
        meta_valid["DrugId"].to_numpy(),
    )
    q80, q90 = apply_residual_quantiles(
        preds_valid_cal,
        meta_valid["DrugId"].to_numpy(),
        q_map,
    )
    valid_preview["y_pred_q80"] = q80
    valid_preview["y_pred_q90"] = q90
    valid_preview["y_pred_low"] = pred_low
    valid_preview["y_pred_mid"] = pred_mid
    valid_preview["y_pred_high"] = pred_high
    valid_preview["error_rate"] = error_rate_series.to_numpy()
    valid_preview["y_pred"] = np.round(valid_preview["y_pred"]).astype(int)
    valid_preview["y_pred_low"] = np.round(valid_preview["y_pred_low"]).astype(int)
    valid_preview["y_pred_mid"] = np.round(valid_preview["y_pred_mid"]).astype(int)
    valid_preview["y_pred_high"] = np.round(valid_preview["y_pred_high"]).astype(int)
    valid_preview["y_pred_round"] = valid_preview["y_pred"]
    valid_preview["y_pred_cal_round"] = np.round(valid_preview["y_pred_cal"]).astype(int)
    low_value, mid_value, high_value = compute_decile_values(
        valid_preview["y_pred_low"].to_numpy(),
        valid_preview["y_pred_high"].to_numpy(),
    )
    valid_preview["low_value"] = low_value
    valid_preview["mid_value"] = mid_value
    valid_preview["high_value"] = high_value
    valid_preview["residual"] = valid_preview["y_true"] - valid_preview["y_pred"]
    valid_preview["residual_cal"] = valid_preview["y_true"] - valid_preview["y_pred_cal"]
    run_valid_path = pred_dir / "upw_valid_predictions.csv"
    _write_dual_csv(valid_preview, run_valid_path, processed_dir / run_valid_path.name)

    run_drift_path = diagnostics_dir / "upw_fold_drift.csv"
    monitor_drift(X_train_enc, X_valid_enc, output_path=str(run_drift_path))
    _copy_if_exists(run_drift_path, processed_dir / run_drift_path.name)
    monitor_rolling_rmse(y_valid.reset_index(drop=True), preds_valid_cal)

    # Save model and minimal artifacts for reuse
    run_model_path = active_run_dir / "models" / "upw_xgb_model.json"
    run_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(run_model_path)
    latest_model_path = artifacts_dir / "models" / "upw_xgb_model.json"
    _copy_if_exists(run_model_path, latest_model_path)
    print(f"\nModel saved to {run_model_path}")
    print(
        f"Train size: {len(X_train_enc)}, Validation size: {len(X_valid_enc)}, "
        f"Test size: {len(X_test_enc)}"
    )

    # Baseline comparisons
    baseline_preds = _baseline_predictions(y, meta, ma_window=4, seasonal_period=52)
    train_weeks = set(meta_train["week_start_ts"].unique())
    valid_weeks = set(meta_valid["week_start_ts"].unique())
    test_weeks = set(meta_test["week_start_ts"].unique())
    train_mask = meta["week_start_ts"].isin(train_weeks)
    valid_mask = meta["week_start_ts"].isin(valid_weeks)
    test_mask = meta["week_start_ts"].isin(test_weeks)

    baseline_metrics = {
        "train": {
            "naive": _evaluate_with_mask(y, baseline_preds["pred_naive"], train_mask, "naive_train"),
            "moving_avg": _evaluate_with_mask(
                y, baseline_preds["pred_moving_avg"], train_mask, "moving_avg_train"
            ),
            "seasonal_naive": _evaluate_with_mask(
                y, baseline_preds["pred_seasonal_naive"], train_mask, "seasonal_naive_train"
            ),
        },
        "valid": {
            "naive": _evaluate_with_mask(y, baseline_preds["pred_naive"], valid_mask, "naive_valid"),
            "moving_avg": _evaluate_with_mask(
                y, baseline_preds["pred_moving_avg"], valid_mask, "moving_avg_valid"
            ),
            "seasonal_naive": _evaluate_with_mask(
                y, baseline_preds["pred_seasonal_naive"], valid_mask, "seasonal_naive_valid"
            ),
        },
        "test": {
            "naive": _evaluate_with_mask(y, baseline_preds["pred_naive"], test_mask, "naive_test"),
            "moving_avg": _evaluate_with_mask(
                y, baseline_preds["pred_moving_avg"], test_mask, "moving_avg_test"
            ),
            "seasonal_naive": _evaluate_with_mask(
                y, baseline_preds["pred_seasonal_naive"], test_mask, "seasonal_naive_test"
            ),
        },
    }

    shuffle_metrics = shuffle_target_test(X, y, meta, random_seed=active_settings.random_seed)
    print("\n=== Shuffled-target test ===")
    print(
        f"Valid: RMSE={shuffle_metrics['rmse']:.3f} | R2={shuffle_metrics['r2']:.3f} | "
        f"MAE={shuffle_metrics['mae']:.3f}"
    )

    data_manifest = _build_data_manifest(df, meta_train, meta_valid, meta_test, weekly_features_path)
    train_params = {
        "xgb_params": model_params,
        "split_weeks": {"train": "years_1_2", "valid": "year3_h1", "test": "year3_h2"},
        "num_boost_round": 250,
        "early_stopping_rounds": 35,
        "two_stage": {"enabled": False},
        "weights": {
            "time_decay_half_life_weeks": 26,
            "volume_weight": "sqrt(train_total/median)",
            "censored_weight": 0.5,
        },
        "baseline_windows": {"moving_avg": 4, "seasonal_naive": 52},
        "walk_forward": {
            "enabled": bool(active_run_walk_forward),
            "min_train_weeks": 30,
            "error_window": 8,
        },
    }
    metrics_payload = {
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
        "valid_calibrated": valid_cal_metrics,
        "test_calibrated": test_cal_metrics,
        "cv": {"folds": cv_metrics, "avg_rmse": avg_rmse, "avg_mae": avg_mae},
        "walk_forward": wf_metrics,
        "baselines": baseline_metrics,
        "shuffle_test": shuffle_metrics,
    }

    write_json(active_run_dir / "data_manifest.json", data_manifest)
    write_json(active_run_dir / "train_params.json", train_params)
    write_json(active_run_dir / "metrics.json", metrics_payload)
    metrics_dir = artifacts_dir / "metrics" / active_run_dir.name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "metrics.json", metrics_payload)
    return metrics_payload


if __name__ == "__main__":
    main()
