"""Drift monitoring, leakage auditing, and diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def monitor_drift(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_path: str = "data/processed/upw_fold_drift.csv",
) -> pd.DataFrame:
    """Basic drift check: compare means/stds of numeric features between train and valid."""
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    rows: list[dict] = []
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
    logger.info("=== Fold drift check (top 5 by mean %% diff) ===")
    for _, row in top5.iterrows():
        logger.info(
            "%s: train_mean=%.3f, valid_mean=%.3f, mean_pct_diff=%.2f%%",
            row["feature"],
            row["train_mean"],
            row["valid_mean"],
            row["mean_pct"] * 100,
        )
    alerts = drift_df[drift_df["mean_pct"].abs() > 0.2]
    if not alerts.empty:
        logger.warning("Potential drift on feature means (>20%% diff):")
        for _, row in alerts.iterrows():
            logger.warning(
                " - %s: mean_pct_diff=%.2f%%", row["feature"], row["mean_pct"] * 100
            )
    return drift_df


def monitor_rolling_rmse(
    y_true: pd.Series,
    y_pred: np.ndarray,
    window: int = 8,
) -> None:
    """Simple rolling RMSE monitor; alerts if last exceeds 2x historical median."""
    residuals = (y_true - y_pred) ** 2
    rmse_roll = np.sqrt(residuals.rolling(window=window, min_periods=window).mean())
    if rmse_roll.dropna().empty:
        return
    median_hist = rmse_roll.median()
    last_rmse = rmse_roll.dropna().iloc[-1]
    logger.info("Rolling RMSE (window=%d) last=%.3f, median=%.3f", window, last_rmse, median_hist)
    if last_rmse > 2 * median_hist:
        logger.warning("Rolling RMSE exceeds 2x historical median. Possible performance drift.")


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
    rows: list[dict] = []

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
        g["spike_score_8_chk"] = (lag_series - g["UPW_rollmean_8_chk"]) / (
            g["UPW_rollstd_8_chk"] + epsilon
        )
        g["diff_rate_chk"] = g["diff_rate_chk"].clip(lower=-3, upper=3)
        g["Z_score_chk"] = g["Z_score_chk"].clip(lower=-3, upper=3)
        regime_raw = ((g["week_of_year"] <= 8) | (g["week_of_year"] >= 45)).astype(float)
        regime_lag = regime_raw.shift(1)
        g["regime_prob_8w_chk"] = regime_lag.rolling(window=8, min_periods=4).mean()
        g["spike_flag_chk"] = (
            (lag_series > 1.5 * (g["UPW_rollmean_12_chk"] + epsilon))
            | (g["diff_rate_chk"] > 1.0)
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
    logger.info(
        "Walk-forward Fold1 summary: Week_ts=%d | Drugs=%d | New drugs=%d | Top abs_err=%.3f",
        int(first_week),
        fold1["DrugId"].nunique(),
        int(fold1["is_new_drug"].sum()),
        fold1["abs_err"].max(),
    )
    return fold1
