"""Tests for weighting module."""

import numpy as np
import pandas as pd

from pharmacystore.models.weighting import compute_sample_weights


def test_compute_sample_weights_basic():
    meta = pd.DataFrame({
        "week_start_ts": [1, 2, 3, 4, 5, 6, 7, 8],
        "DrugId": [1, 1, 1, 1, 2, 2, 2, 2],
    })
    y = pd.Series([10, 20, 30, 40, 5, 10, 15, 20])
    train_mask = np.array([True, True, True, True, True, True, False, False])
    
    weights = compute_sample_weights(meta, y, train_mask)
    
    assert len(weights) == len(meta)
    assert np.all(weights > 0)
    assert weights[-1] > weights[0]  # Recent weeks should have higher weight


def test_compute_sample_weights_empty():
    meta = pd.DataFrame()
    y = pd.Series([])
    train_mask = np.array([])
    
    weights = compute_sample_weights(meta, y, train_mask)
    assert len(weights) == 0


def test_compute_sample_weights_with_censored():
    meta = pd.DataFrame({
        "week_start_ts": [1, 2, 3, 4],
        "DrugId": [1, 1, 1, 1],
        "censored_flag": [0, 1, 0, 0],
    })
    y = pd.Series([10, 20, 30, 40])
    train_mask = np.array([True, True, True, False])
    
    weights = compute_sample_weights(meta, y, train_mask)
    
    assert weights[1] < weights[0]  # Censored row should have lower weight
