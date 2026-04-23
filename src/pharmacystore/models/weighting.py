"""Sample weighting utilities for model training."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    weeks_sorted = np.sort(meta["week_start_ts"].unique())
    week_rank = {wk: i for i, wk in enumerate(weeks_sorted)}
    max_rank = max(week_rank.values()) if week_rank else 0
    decay = np.log(2) / max(half_life_weeks, 1)
    time_weight = meta["week_start_ts"].map(
        lambda wk: float(np.exp(decay * (week_rank[wk] - max_rank)))
    ).to_numpy()

    train_meta = meta.loc[train_mask].reset_index(drop=True)
    train_y = y.loc[train_mask].reset_index(drop=True)
    drug_totals = train_y.groupby(train_meta["DrugId"]).sum()
    median_total = float(drug_totals.median()) if not drug_totals.empty else 1.0
    drug_scale = (drug_totals / max(median_total, 1e-6)) ** 0.5
    drug_scale = drug_scale.clip(lower=min_drug_weight, upper=max_drug_weight)
    drug_weight = meta["DrugId"].map(drug_scale).fillna(1.0).to_numpy()

    weights = time_weight * drug_weight

    for col in ["censored_flag", "stockout_flag", "is_censored"]:
        if col in meta.columns:
            weights = np.where(meta[col].fillna(0).to_numpy() > 0, weights * 0.5, weights)
            break

    return weights
