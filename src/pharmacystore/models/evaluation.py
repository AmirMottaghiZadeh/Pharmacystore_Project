"""Model evaluation metrics and baseline comparisons."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: pd.Series,
    preds: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """Compute a comprehensive set of regression metrics."""
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(preds, dtype=float)
    valid = np.isfinite(y_arr) & np.isfinite(p_arr)
    y_arr = y_arr[valid]
    p_arr = p_arr[valid]
    if len(y_arr) == 0:
        logger.warning("evaluate_predictions(%s): no valid observations.", label)
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "median_ae": float("nan"),
            "r2": float("nan"),
            "bias": float("nan"),
            "wape": float("nan"),
            "wwape": float("nan"),
            "smape": float("nan"),
            "under_forecast_rate": float("nan"),
            "under_forecast_share": float("nan"),
        }
    rmse = float(np.sqrt(mean_squared_error(y_arr, p_arr)))
    mae = float(mean_absolute_error(y_arr, p_arr))
    median_ae = float(median_absolute_error(y_arr, p_arr))
    r2 = float(r2_score(y_arr, p_arr))
    bias = float(np.mean(p_arr - y_arr))
    eps = 1e-6
    wape = float(np.sum(np.abs(y_arr - p_arr)) / (np.sum(np.abs(y_arr)) + eps))
    weights_vol = np.abs(y_arr) + eps
    wwape = float(
        np.sum(weights_vol * np.abs(y_arr - p_arr)) / (np.sum(weights_vol * np.abs(y_arr)) + eps)
    )
    smape = float(
        np.mean(2 * np.abs(y_arr - p_arr) / (np.abs(y_arr) + np.abs(p_arr) + eps)) * 100
    )
    under_mask = p_arr < y_arr
    under_forecast_rate = float(np.mean(under_mask))
    under_total = float(np.sum(np.abs(y_arr[under_mask] - p_arr[under_mask])))
    total_err = float(np.sum(np.abs(y_arr - p_arr)))
    under_forecast_share = under_total / (total_err + eps)
    return {
        "rmse": rmse,
        "mae": mae,
        "median_ae": median_ae,
        "r2": r2,
        "bias": bias,
        "wape": wape,
        "wwape": wwape,
        "smape": smape,
        "under_forecast_rate": under_forecast_rate,
        "under_forecast_share": under_forecast_share,
    }


def evaluate_with_mask(
    y: pd.Series,
    preds: np.ndarray,
    mask: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """Evaluate predictions on a boolean-masked subset."""
    count = int(mask.sum())
    if count < 2:
        return {
            "n_obs": count,
            "rmse": None,
            "mae": None,
            "median_ae": None,
            "r2": None,
            "bias": None,
            "wape": None,
            "smape": None,
        }
    metrics = evaluate_predictions(y[mask], preds[mask], label=label)
    metrics["n_obs"] = count
    return metrics


def baseline_predictions(
    y: pd.Series,
    meta: pd.DataFrame,
    ma_window: int = 4,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    """Generate naive, moving-average, and seasonal-naive baseline predictions."""
    data = pd.DataFrame(
        {
            "y": y.values,
            "DrugId": meta["DrugId"].values,
            "week_start_ts": meta["week_start_ts"].values,
        }
    ).sort_values(["DrugId", "week_start_ts"])

    data["pred_naive"] = data.groupby("DrugId")["y"].shift(1)
    data["pred_moving_avg"] = (
        data.groupby("DrugId")["y"]
        .transform(lambda s: s.shift(1).rolling(window=ma_window, min_periods=1).mean())
    )
    data["pred_seasonal_naive"] = data.groupby("DrugId")["y"].shift(seasonal_period)

    for col in ["pred_naive", "pred_moving_avg", "pred_seasonal_naive"]:
        data[col] = data[col].fillna(data["y"].median())
        data[col] = data[col].clip(lower=0)

    return data.reindex(y.index)


def build_error_band(
    y_pred: np.ndarray,
    error_rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (low, mid, high) bands around predictions based on error rates."""
    pred = np.asarray(y_pred, dtype=float)
    rate = np.asarray(error_rate, dtype=float)
    low = np.clip(pred * (1 - rate), 0, None)
    mid = pred.copy()
    high = pred * (1 + rate)
    return low, mid, high


def compute_decile_values(
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert low/high bands to purchase-quantity decile values."""
    low_arr = np.asarray(low, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    mid_arr = (low_arr + high_arr) / 2
    return np.round(low_arr).astype(int), np.round(mid_arr).astype(int), np.round(high_arr).astype(int)


def print_evaluation(
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
    test_metrics: dict[str, float] | None = None,
) -> None:
    """Pretty-print metrics for train/validation/test splits."""

    def _fmt(m: dict[str, float]) -> str:
        return (
            f"RMSE={m['rmse']:.3f} | MAE={m['mae']:.3f} | MedAE={m['median_ae']:.3f} | "
            f"R2={m['r2']:.3f} | WAPE={m['wape']:.3f} | WWAPE={m.get('wwape', 0):.3f} | "
            f"sMAPE={m['smape']:.3f} | Bias={m['bias']:.3f} | "
            f"UnderRate={m.get('under_forecast_rate', 0):.2f} | "
            f"UnderShare={m.get('under_forecast_share', 0):.3f}"
        )

    logger.info("=== Evaluation ===")
    logger.info("Train: %s", _fmt(train_metrics))
    logger.info("Valid: %s", _fmt(valid_metrics))
    if test_metrics is not None:
        logger.info("Test:  %s", _fmt(test_metrics))


def shuffle_target_test(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    num_boost_round: int = 120,
    early_stopping_rounds: int = 20,
    random_seed: int | None = None,
) -> dict[str, float]:
    """Leakage check: shuffle targets and retrain; metrics should be near-zero signal."""
    import xgboost as xgb

    from pharmacystore.models.encoding import encode_categoricals
    from pharmacystore.models.splitting import ensure_week_start_ts

    meta = ensure_week_start_ts(meta)
    order = np.argsort(meta["week_start_ts"].values)
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

    params: dict = {
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
