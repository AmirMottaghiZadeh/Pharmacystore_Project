"""Walk-forward (rolling-origin) backtesting."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from pharmacystore.models.encoding import apply_label_encoders, encode_categoricals
from pharmacystore.models.splitting import ensure_week_start_ts
from pharmacystore.models.weighting import compute_sample_weights

logger = logging.getLogger(__name__)


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

    meta = ensure_week_start_ts(meta)
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

    wf_df = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["week_start_ts", "DrugId"])
        .reset_index(drop=True)
    )
    eps = 1e-6
    min_periods = max(2, error_window // 2)
    wf_df["rolling_mae"] = (
        wf_df.groupby("DrugId")["abs_err"]
        .apply(
            lambda s: s.rolling(window=error_window, min_periods=min_periods).mean().shift(1)
        )
        .reset_index(level=0, drop=True)
    )
    wf_df["rolling_mean_y"] = (
        wf_df.groupby("DrugId")["y_true"]
        .apply(
            lambda s: s.rolling(window=error_window, min_periods=min_periods).mean().shift(1)
        )
        .reset_index(level=0, drop=True)
    )
    wf_df["error_rate_roll"] = wf_df["rolling_mae"] / (wf_df["rolling_mean_y"].abs() + eps)
    global_error_rate = float(wf_df["abs_err"].mean() / (wf_df["y_true"].abs().mean() + eps))
    global_error_rate = float(np.clip(global_error_rate, 0.0, 1.0))
    wf_df["error_rate"] = wf_df["error_rate_roll"].fillna(global_error_rate)
    wf_df["error_rate"] = wf_df["error_rate"].clip(lower=0.0, upper=1.0)

    from pharmacystore.models.evaluation import evaluate_predictions

    wf_metrics = evaluate_predictions(
        pd.Series(wf_df["y_true"].values),
        wf_df["y_pred"].values,
        label="walk_forward",
    )
    wf_metrics["n_weeks"] = int(wf_df["week_start_ts"].nunique())
    wf_metrics["n_rows"] = len(wf_df)

    wf_df.to_csv(output_path, index=False)
    logger.info(
        "Walk-forward complete: weeks=%d, rows=%d, RMSE=%.3f, MAE=%.3f",
        wf_metrics["n_weeks"],
        wf_metrics["n_rows"],
        wf_metrics["rmse"],
        wf_metrics["mae"],
    )
    return wf_df, wf_metrics
