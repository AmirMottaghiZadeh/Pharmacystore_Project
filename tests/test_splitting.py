"""Tests for splitting module."""

import numpy as np
import pandas as pd
import pytest

from pharmacystore.models.splitting import ensure_week_start_ts, split_weeks, three_year_split


def test_ensure_week_start_ts_already_present():
    meta = pd.DataFrame({"week_start_ts": [1, 2, 3]})
    result = ensure_week_start_ts(meta)
    assert "week_start_ts" in result.columns
    assert result["week_start_ts"].tolist() == [1, 2, 3]


def test_ensure_week_start_ts_derives_from_week_start():
    meta = pd.DataFrame({"week_start": ["2023-01-02", "2023-01-09"]})
    result = ensure_week_start_ts(meta)
    assert "week_start_ts" in result.columns
    assert result["week_start_ts"].dtype == np.int64


def test_ensure_week_start_ts_raises_without_columns():
    meta = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(KeyError):
        ensure_week_start_ts(meta)


def test_split_weeks_basic():
    meta = pd.DataFrame({"week_start_ts": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    train, valid, test = split_weeks(meta, train_frac=0.6, valid_frac=0.2)
    assert len(train) == 6
    assert len(valid) == 2
    assert len(test) == 2


def test_split_weeks_insufficient_data():
    meta = pd.DataFrame({"week_start_ts": [1, 2]})
    with pytest.raises(ValueError, match="at least 3 unique weeks"):
        split_weeks(meta)


def test_three_year_split_basic():
    dates = pd.date_range("2020-01-06", periods=156, freq="W-MON")
    meta = pd.DataFrame({
        "week_start": dates,
        "week_start_ts": (dates.astype("int64") // 10**9).tolist()
    })
    train, valid, test = three_year_split(meta)
    assert len(train) > 0
    assert len(valid) > 0
    assert len(test) > 0
