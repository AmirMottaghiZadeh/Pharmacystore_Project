"""Time-based dataset splitting utilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_week_start_ts(meta: pd.DataFrame) -> pd.DataFrame:
    """Ensure *meta* contains a ``week_start_ts`` column (epoch seconds)."""
    if "week_start_ts" in meta.columns:
        return meta
    if "week_start" not in meta.columns:
        raise KeyError("week_start_ts missing from meta and cannot derive from week_start.")
    meta = meta.copy()
    meta["week_start_ts"] = pd.to_datetime(meta["week_start"]).astype("int64") // 10**9
    return meta


def split_weeks(
    meta: pd.DataFrame,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple fractional split based on sorted unique ``week_start_ts``."""
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


def three_year_split(
    meta: pd.DataFrame,
    min_valid_weeks: int = 4,
    min_test_weeks: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Year-based split: years 1-2 train, year 3 H1 valid, year 3 H2 test."""
    weeks_df = (
        meta[["week_start_ts", "week_start"]]
        .drop_duplicates()
        .sort_values("week_start_ts")
    )
    if weeks_df.empty:
        raise ValueError("No weekly information available for splitting.")

    week_dates = pd.to_datetime(weeks_df["week_start"])
    base_year = int(week_dates.min().year)
    weeks_df["rel_year"] = week_dates.dt.year - base_year + 1
    weeks_df["month"] = week_dates.dt.month

    train_weeks = weeks_df.loc[
        weeks_df["rel_year"].isin([1, 2]), "week_start_ts"
    ].to_numpy()
    valid_weeks = weeks_df.loc[
        (weeks_df["rel_year"] == 3) & (weeks_df["month"] <= 6), "week_start_ts"
    ].to_numpy()
    test_weeks = weeks_df.loc[
        (weeks_df["rel_year"] == 3) & (weeks_df["month"] >= 7), "week_start_ts"
    ].to_numpy()

    if (
        len(valid_weeks) < min_valid_weeks
        or len(test_weeks) < min_test_weeks
        or len(train_weeks) == 0
    ):
        logger.warning(
            "Three-year split did not produce enough weeks; falling back to fractional split."
        )
        return split_weeks(meta, train_frac=0.7, valid_frac=0.15)

    return train_weeks, valid_weeks, test_weeks
