"""Tests for evaluation module."""

import numpy as np
import pandas as pd

from pharmacystore.models.evaluation import (
    build_error_band,
    compute_decile_values,
    evaluate_predictions,
)


def test_evaluate_predictions_perfect():
    y_true = pd.Series([10, 20, 30, 40])
    preds = np.array([10, 20, 30, 40])
    
    metrics = evaluate_predictions(y_true, preds, label="test")
    
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["bias"] == 0.0


def test_evaluate_predictions_with_error():
    y_true = pd.Series([10, 20, 30, 40])
    preds = np.array([12, 18, 32, 38])
    
    metrics = evaluate_predictions(y_true, preds, label="test")
    
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert 0 < metrics["r2"] < 1


def test_build_error_band():
    y_pred = np.array([100, 200, 300])
    error_rate = np.array([0.1, 0.2, 0.15])
    
    low, mid, high = build_error_band(y_pred, error_rate)
    
    assert np.allclose(low, [90, 160, 255])
    assert np.allclose(mid, [100, 200, 300])
    assert np.allclose(high, [110, 240, 345])


def test_compute_decile_values():
    low = np.array([90.5, 160.3, 255.7])
    high = np.array([110.2, 240.8, 345.1])
    
    low_val, mid_val, high_val = compute_decile_values(low, high)
    
    assert low_val.dtype == np.int64
    assert mid_val.dtype == np.int64
    assert high_val.dtype == np.int64
    assert np.all(low_val <= mid_val)
    assert np.all(mid_val <= high_val)
