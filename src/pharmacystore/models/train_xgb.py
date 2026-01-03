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

    # Seasonality encoding with normalized sin/cos
    week_num = data["week_of_year"]
    data["week_sin"] = np.sin(2 * np.pi * week_num / 52)
    data["week_cos"] = np.cos(2 * np.pi * week_num / 52)
    if "week_sinus" in data.columns:
        data = data.drop(columns=["week_sinus"])

    # Ensure lagged target exists (built upstream, but recompute defensively)
    if "UPW_lag1" not in data.columns:
        data["UPW_lag1"] = data.groupby("DrugId")["UPW"].shift(1)

    # Trend features strictly on lagged series (no UPW_t leakage)
    lag_series = data["UPW_lag1"]
    short_roll = lag_series.rolling(window=4, min_periods=4).mean()
    long_roll = lag_series.rolling(window=12, min_periods=12).mean()
    data["UPW_rollmean_12"] = long_roll
    data["trend_short_vs_long"] = short_roll - long_roll
    data["trend_slope_8w"] = lag_series.rolling(window=8, min_periods=8).mean().diff()
    data["trend_accel"] = data["trend_slope_8w"].diff()

    # Regime flag for start/end of year (helps separate regimes)
    data["regime_flag"] = ((week_num <= 8) | (week_num >= 45)).astype(int)

    # Drop rows without sufficient lag history
    data = data.replace([np.inf, -np.inf], np.nan)
    history_cols = [
        col
        for col in [
            "UPW_lag1",
            "UPW_rollmean_2",
            "UPW_rollmean_4",
            "UPW_rollmean_6",
            "UPW_rollstd_3",
            "UPW_rollstd_6",
            "UPW_rollmin",
            "UPW_rollmax",
            "diff_rate",
            "avg_point",
            "Z_score",
            "UPW_rollmean_12",
            "trend_short_vs_long",
            "trend_slope_8w",
            "trend_accel",
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
            "regime_flag",
            "quarter",
        ]
        if col in data.columns
    ]
    meta = data[meta_cols].copy()

    # Target and features
    y = data["UPW"].astype(float)
    # Drop target and any aggregates/categories derived from full-history UPW to avoid leakage
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
    ]
    X = data.drop(columns=[c for c in drop_target_like if c in data.columns])

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
    """Train/validate an XGBoost regressor using a time-based split."""
    # Time-based ordering to avoid leakage
    order = np.argsort(X["week_start_ts"].values)
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)
    meta_ordered = meta.iloc[order].reset_index(drop=True)
    weights_ordered = np.linspace(0.7, 1.3, len(X_ordered))

    train_weeks, valid_weeks, test_weeks = _split_weeks(
        meta_ordered, train_frac=train_frac, valid_frac=valid_frac
    )
    train_mask = meta_ordered["week_start_ts"].isin(train_weeks).to_numpy()
    valid_mask = meta_ordered["week_start_ts"].isin(valid_weeks).to_numpy()
    test_mask = meta_ordered["week_start_ts"].isin(test_weeks).to_numpy()

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
        "max_depth": 4,  # shallower tree to reduce variance
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 6.0,  # higher to control overfitting
        "gamma": 0.4,  # minimum loss reduction to make a split
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
    }
    if random_seed is not None:
        active_params.setdefault("seed", int(random_seed))

    evals = [(train_dm, "train"), (valid_dm, "valid")]
    model = xgb.train(
        params=active_params,
        dtrain=train_dm,
        num_boost_round=300,  # ~250-300 as requested
        evals=evals,
        early_stopping_rounds=40,
        verbose_eval=50,
    )

    preds_train = model.predict(train_dm)
    preds_valid = model.predict(valid_dm)
    preds_test = model.predict(test_dm)

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
    n_splits: int = 3,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 40,
) -> list[dict[str, float]]:
    """Perform time-series cross-validation and return metrics per fold."""
    order = np.argsort(X["week_start_ts"].values)
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)
    week_vals = X_ordered["week_start_ts"].to_numpy()
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

        X_train, X_valid = X_ordered.loc[train_mask], X_ordered.loc[valid_mask]
        y_train, y_valid = y_ordered.loc[train_mask], y_ordered.loc[valid_mask]
        weights = np.linspace(0.7, 1.3, len(X_ordered))
        w_train, w_valid = weights[train_mask], weights[valid_mask]

        X_train_enc, X_valid_enc, _ = encode_categoricals(X_train, X_valid)

        train_dm = xgb.DMatrix(X_train_enc, label=y_train, weight=w_train)
        valid_dm = xgb.DMatrix(X_valid_enc, label=y_valid, weight=w_valid)
        evals = [(train_dm, "train"), (valid_dm, "valid")]

        booster = xgb.train(
            params=params,
            dtrain=train_dm,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        preds = booster.predict(valid_dm)
        metrics = evaluate_predictions(y_valid, preds, label=f"fold_{fold}")
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
    regime_col: str = "regime_flag",
    recent_window: int = 8,
) -> np.ndarray:
    """Bias-correct predictions per regime; no global bias fallback."""
    bias_map: dict = {}
    residuals_train = preds_train - y_train
    if regime_col in meta_train.columns:
        for regime, idx in meta_train.groupby(regime_col).groups.items():
            bias_map[regime] = float(np.mean(residuals_train[idx]))

    recent_bias = float(residuals_train.tail(recent_window).mean()) if len(residuals_train) else 0.0

    if regime_col in meta_valid.columns and bias_map:
        adj = [
            bias_map.get(meta_valid.iloc[i][regime_col], recent_bias)
            for i in range(len(meta_valid))
        ]
        return preds + np.array(adj)

    return preds + recent_bias


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
    """Rolling-origin training: for each week, fit on past weeks and track rolling error bands."""
    if params is None:
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

    order = np.argsort(X["week_start_ts"].values)
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)
    meta_ordered = meta.iloc[order].reset_index(drop=True)
    weights_ordered = np.linspace(0.7, 1.3, len(X_ordered))

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
        meta_train_all = meta_ordered.loc[train_mask].reset_index(drop=True)
        w_train_all = weights_ordered[train_mask]

        inner_split = max(int(len(X_train_all) * 0.85), len(X_train_all) - 150)
        inner_split = min(max(inner_split, 1), len(X_train_all) - 1)
        X_inner_train_raw = X_train_all.iloc[:inner_split]
        X_inner_valid_raw = X_train_all.iloc[inner_split:]
        y_inner_train = y_train_all.iloc[:inner_split]
        y_inner_valid = y_train_all.iloc[inner_split:]
        w_inner_train = w_train_all[:inner_split]
        w_inner_valid = w_train_all[inner_split:]

        X_inner_train, X_inner_valid, encoders = encode_categoricals(X_inner_train_raw, X_inner_valid_raw)
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

        # Bias calibration uses full in-sample residuals
        train_full_enc = apply_label_encoders(X_train_all, encoders)
        train_full_dm = xgb.DMatrix(train_full_enc, label=y_train_all, weight=w_train_all)
        preds_train_full = booster.predict(train_full_dm)

        X_valid_raw = X_ordered.loc[valid_mask].reset_index(drop=True)
        y_valid = y_ordered.loc[valid_mask].reset_index(drop=True)
        meta_valid = meta_ordered.loc[valid_mask].reset_index(drop=True)
        X_valid_enc = apply_label_encoders(X_valid_raw, encoders)
        valid_dm = xgb.DMatrix(X_valid_enc)

        preds_valid = booster.predict(valid_dm)
        preds_valid_cal = apply_regime_calibration(
            preds_valid, meta_valid, y_train_all, preds_train_full, meta_train_all
        )

        week_df = meta_valid.copy()
        if "week_start_ts" not in week_df.columns and "week_start_ts" in X_valid_raw.columns:
            week_df["week_start_ts"] = X_valid_raw["week_start_ts"].values
        week_df["y_true"] = y_valid.values
        week_df["y_pred"] = preds_valid
        week_df["y_pred_cal"] = preds_valid_cal
        week_df["abs_err"] = np.abs(week_df["y_pred_cal"] - week_df["y_true"])
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
    wf_df["error_rate"] = wf_df["error_rate_roll"].fillna(global_error_rate)

    pred_low, pred_mid, pred_high = build_error_band(
        wf_df["y_pred_cal"].to_numpy(), wf_df["error_rate"].to_numpy()
    )
    wf_df["y_pred_low"] = pred_low
    wf_df["y_pred_mid"] = pred_mid
    wf_df["y_pred_high"] = pred_high

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
    metrics = {
        "label": label,
        "rmse": rmse,
        "mae": mae,
        "median_ae": medae,
        "r2": r2,
        "bias": bias,
        "wape": wape,
        "smape": smape,
    }
    return metrics


def build_error_band(
    preds: np.ndarray, error_rate: float | np.ndarray | pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (low, mid, high) predictions using a symmetric percentage band."""
    rates = np.asarray(error_rate)
    if np.any(rates < 0):
        raise ValueError("error_rate must be non-negative")
    mid = preds
    delta = np.abs(preds) * rates
    low = np.clip(mid - delta, 0, None)
    high = mid + delta
    return low, mid, high


def shuffle_target_test(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    num_boost_round: int = 120,
    early_stopping_rounds: int = 20,
    random_seed: int | None = None,
) -> dict[str, float]:
    """Leakage check: shuffle targets and retrain; metrics should be near-zero signal."""
    order = np.argsort(X["week_start_ts"].values)
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
            f"R2={m['r2']:.3f} | WAPE={m['wape']:.3f} | sMAPE={m['smape']:.3f} | "
            f"Bias={m['bias']:.3f}"
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
    model_params = {
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

    train_metrics = evaluate_predictions(y_train, preds_train, label="train")
    valid_metrics = evaluate_predictions(y_valid, preds_valid, label="valid")
    test_metrics = evaluate_predictions(y_test, preds_test, label="test")
    preds_valid_cal = apply_regime_calibration(
        preds_valid, meta_valid, y_train, preds_train, meta_train, regime_col="regime_flag"
    )
    preds_test_cal = apply_regime_calibration(
        preds_test, meta_test, y_train, preds_train, meta_train, regime_col="regime_flag"
    )
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
    global_error_rate = float(abs_err.mean() / (np.abs(y_valid).mean() + eps))
    error_rate_series = meta_valid["DrugId"].map(
        per_drug.set_index("DrugId")["error_rate"]
    ).fillna(global_error_rate)
    pred_low, pred_mid, pred_high = build_error_band(preds_valid_cal, error_rate_series.to_numpy())
    valid_cal_metrics = evaluate_predictions(y_valid, preds_valid_cal, label="valid_calibrated")
    test_cal_metrics = evaluate_predictions(y_test, preds_test_cal, label="test_calibrated")
    print_evaluation(train_metrics, valid_metrics, test_metrics)
    print("Calibrated (bias-corrected) valid metrics:")
    print(
        f"RMSE={valid_cal_metrics['rmse']:.3f} | MAE={valid_cal_metrics['mae']:.3f} | "
        f"MedAE={valid_cal_metrics['median_ae']:.3f} | R2={valid_cal_metrics['r2']:.3f} | "
        f"WAPE={valid_cal_metrics['wape']:.3f} | sMAPE={valid_cal_metrics['smape']:.3f} | "
        f"Bias={valid_cal_metrics['bias']:.3f}"
    )
    print("Calibrated (bias-corrected) test metrics:")
    print(
        f"RMSE={test_cal_metrics['rmse']:.3f} | MAE={test_cal_metrics['mae']:.3f} | "
        f"MedAE={test_cal_metrics['median_ae']:.3f} | R2={test_cal_metrics['r2']:.3f} | "
        f"WAPE={test_cal_metrics['wape']:.3f} | sMAPE={test_cal_metrics['smape']:.3f} | "
        f"Bias={test_cal_metrics['bias']:.3f}"
    )

    # Time-series cross-validation summary
    print("\n=== Time-series cross-validation (rolling splits) ===")
    cv_metrics = time_series_cv(
        X,
        y,
        params=model_params,
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
    valid_preview["y_pred_low"] = pred_low
    valid_preview["y_pred_mid"] = pred_mid
    valid_preview["y_pred_high"] = pred_high
    valid_preview["error_rate"] = error_rate_series.to_numpy()
    valid_preview["y_pred_round"] = np.round(preds_valid).astype(int)
    valid_preview["y_pred_cal_round"] = np.round(preds_valid_cal).astype(int)
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
        "split_weeks": {"train": 0.7, "valid": 0.15, "test": 0.15},
        "num_boost_round": 300,
        "early_stopping_rounds": 40,
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
    return metrics_payload


if __name__ == "__main__":
    main()
