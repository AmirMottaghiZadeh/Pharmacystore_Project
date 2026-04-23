"""Post-training calibration, spike policies, and shock-guard logic."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calibrate_per_drug_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drug_ids: np.ndarray,
    clip_range: tuple[float, float] = (0.5, 2.0),
) -> dict:
    """Compute per-drug scaling factors via least-squares."""
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
    """Apply per-drug calibration scaling."""
    scaled = np.array(y_pred, copy=True)
    for idx, drug in enumerate(drug_ids):
        scaled[idx] = scaled[idx] * scales.get(drug, 1.0)
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
    """Add per-drug residual quantiles to predictions for q80/q90 bands."""
    q80 = np.array(y_pred, copy=True)
    q90 = np.array(y_pred, copy=True)
    for idx, drug in enumerate(drug_ids):
        q_vals = q_map.get(drug, (0.0, 0.0))
        q80[idx] = y_pred[idx] + q_vals[0]
        q90[idx] = y_pred[idx] + q_vals[1]
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

    z_thr: dict = {}
    for drug in np.unique(drug_ids):
        vals = spike_score[drug_ids == drug]
        if len(vals) == 0:
            z_thr[drug] = min_z
        else:
            z_thr[drug] = max(min_z, float(np.quantile(vals, 0.9)))

    best_k = list(k_values)[0]
    if y_true is not None:
        best_wape = float("inf")
        for k in k_values:
            adjusted = np.array(y_pred, copy=True)
            for idx, drug in enumerate(drug_ids):
                if spike_score[idx] > z_thr.get(drug, min_z):
                    adjusted[idx] = max(adjusted[idx], rollmax_8[idx] * k)
            wape = float(np.sum(np.abs(y_true - adjusted)) / (np.sum(np.abs(y_true)) + 1e-6))
            if wape < best_wape:
                best_wape = wape
                best_k = k

    adjusted = np.array(y_pred, copy=True)
    for idx, drug in enumerate(drug_ids):
        if spike_score[idx] > z_thr.get(drug, min_z):
            adjusted[idx] = max(adjusted[idx], rollmax_8[idx] * best_k)
    return np.clip(adjusted, 0, None)


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
