"""Tests for calibration module."""

import numpy as np
import pandas as pd

from pharmacystore.models.calibration import (
    apply_per_drug_scale,
    calibrate_per_drug_scale,
    compute_residual_quantiles,
    apply_residual_quantiles,
)


def test_calibrate_per_drug_scale():
    y_true = np.array([10, 20, 30, 15, 25, 35])
    y_pred = np.array([8, 16, 24, 12, 20, 28])
    drug_ids = np.array([1, 1, 1, 2, 2, 2])
    
    scales = calibrate_per_drug_scale(y_true, y_pred, drug_ids)
    
    assert 1 in scales
    assert 2 in scales
    assert 0.5 <= scales[1] <= 2.0
    assert 0.5 <= scales[2] <= 2.0


def test_apply_per_drug_scale():
    y_pred = np.array([10, 20, 30, 40])
    drug_ids = np.array([1, 1, 2, 2])
    scales = {1: 1.2, 2: 0.8}
    
    scaled = apply_per_drug_scale(y_pred, drug_ids, scales)
    
    assert np.allclose(scaled[:2], [12, 24])
    assert np.allclose(scaled[2:], [24, 32])


def test_compute_residual_quantiles():
    y_true = np.array([10, 20, 30, 15, 25, 35])
    y_pred = np.array([8, 18, 28, 13, 23, 33])
    drug_ids = np.array([1, 1, 1, 2, 2, 2])
    
    q_map = compute_residual_quantiles(y_true, y_pred, drug_ids, quantiles=(0.5, 0.9))
    
    assert 1 in q_map
    assert 2 in q_map
    assert len(q_map[1]) == 2
    assert len(q_map[2]) == 2


def test_apply_residual_quantiles():
    y_pred = np.array([10, 20, 30])
    drug_ids = np.array([1, 1, 2])
    q_map = {1: (2.0, 5.0), 2: (1.0, 3.0)}
    
    q80, q90 = apply_residual_quantiles(y_pred, drug_ids, q_map)
    
    assert np.allclose(q80, [12, 22, 31])
    assert np.allclose(q90, [15, 25, 33])
