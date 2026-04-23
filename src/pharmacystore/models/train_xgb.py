"""Train an XGBoost model to forecast weekly UPW using all engineered features."""

from __future__ import annotations

import datetime as dt
import logging
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from pharmacystore.config import Settings, get_settings
from pharmacystore.logging_config import setup_logging
from pharmacystore.models.calibration import (
    apply_per_drug_scale,
    apply_residual_quantiles,
    apply_spike_policy,
    calibrate_per_drug_scale,
    compute_residual_quantiles,
)
from pharmacystore.models.encoding import apply_label_encoders, encode_categoricals, prepare_features
from pharmacystore.models.evaluation import (
    baseline_predictions,
    build_error_band,
    compute_decile_values,
    evaluate_predictions,
    evaluate_with_mask,
    print_evaluation,
    shuffle_target_test,
)
from pharmacystore.models.monitoring import (
    audit_feature_leakage,
    monitor_drift,
    monitor_rolling_rmse,
    summarize_walkforward_fold1,
)
from pharmacystore.models.splitting import ensure_week_start_ts, three_year_split
from pharmacystore.models.walkforward import walk_forward_forecast
from pharmacystore.models.weighting import compute_sample_weights
from pharmacystore.pipeline import run_pipeline
from pharmacystore.run_utils import compute_file_hash, create_run_dir, set_global_seed, write_json

logger = logging.getLogger(__name__)

ENABLE_SPIKE_POLICY = False


def build_dataset(settings: Settings | None = None) -> pd.DataFrame:
    """Build the full weekly feature set via the existing pipeline."""
    weekly_df = run_pipeline(settings=settings)
    return weekly_df


def safe_settings_snapshot(settings: Settings) -> dict:
    """Snapshot settings without exposing password."""
    snapshot = settings.model_dump()
    password = snapshot.pop("sql_password", None)
    snapshot["sql_password_set"] = bool(password)
    return snapshot


def iso_range(series: pd.Series) -> dict[str, str | None]:
    """Return ISO date range (min, max) from a datetime series."""
    if series is None:
        return {"min": None, "max": None}
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "max": None}
    return {
        "min": values.min().date().isoformat(),
        "max": values.max().date().isoformat(),
    }


def write_dual_csv(df: pd.DataFrame, run_path: Path, latest_path: Path) -> None:
    """Write CSV to both run-specific and latest locations."""
    run_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_path, index=False)
    if run_path != latest_path:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(latest_path, index=False)


def copy_if_exists(src: Path, dest: Path) -> None:
    """Copy file if source exists."""
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def build_data_manifest(
    df: pd.DataFrame,
    meta_train: pd.DataFrame,
    meta_valid: pd.DataFrame,
    meta_test: pd.DataFrame,
    weekly_features_path: Path,
) -> dict:
    """Build metadata about the dataset splits."""
    manifest = {
        "weekly_rows": int(len(df)),
        "unique_drugs": int(df["DrugId"].nunique()) if "DrugId" in df.columns else None,
        "unique_weeks": int(df["week_start"].nunique()) if "week_start" in df.columns else None,
        "week_start_range": iso_range(df.get("week_start")),
        "train_rows": int(len(meta_train)),
        "valid_rows": int(len(meta_valid)),
        "test_rows": int(len(meta_test)),
        "train_week_range": iso_range(meta_train.get("week_start")),
        "valid_week_range": iso_range(meta_valid.get("week_start")),
        "test_week_range": iso_range(meta_test.get("week_start")),
    }
    if weekly_features_path.exists():
        manifest["weekly_features_hash_md5"] = compute_file_hash(weekly_features_path)
    return manifest


def train_xgb_model(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    params: dict | None = None,
    random_seed: int | None = None,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
):
    """Champion training: raw target + weights + safe features."""
    meta = ensure_week_start_ts(meta)
    order = np.argsort(meta["week_start_ts"].values)
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)
    meta_ordered = meta.iloc[order].reset_index(drop=True)

    train_weeks, valid_weeks, test_weeks = three_year_split(
        meta_ordered, min_valid_weeks=4, min_test_weeks=4
    )
    train_mask = meta_ordered["week_start_ts"].isin(train_weeks).to_numpy()
    valid_mask = meta_ordered["week_start_ts"].isin(valid_weeks).to_numpy()
    test_mask = meta_ordered["week_start_ts"].isin(test_weeks).to_numpy()

    weights_ordered = compute_sample_weights(meta_ordered, y_ordered, train_mask=train_mask)

    X_train_raw = X_ordered.loc[train_mask].reset_index(drop=True)
    X_valid_raw = X_ordered.loc[valid_mask].reset_index(drop=True)
    X_test_raw = X_ordered.loc[test_mask].reset_index(drop=True)
    y_train = y_ordered.loc[train_mask].reset_index(drop=True)
    y_valid = y_ordered.loc[valid_mask].reset_index(drop=True)
    y_test = y_ordered.loc[test_mask].reset_index(drop=True)
    meta_train = meta_ordered.loc[train_mask].reset_index(drop=True)
    meta_valid = meta_ordered.loc[valid_mask].reset_index(drop=True)
    meta_test = meta_ordered.loc[test_mask].reset_index(drop=True)
    w_train = weights_ordered[train_mask]
    w_valid = weights_ordered[valid_mask]
    w_test = weights_ordered[test_mask]

    X_train, X_valid, encoders = encode_categoricals(X_train_raw, X_valid_raw)
    X_test = apply_label_encoders(X_test_raw, encoders)

    train_dm = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    valid_dm = xgb.DMatrix(X_valid, label=y_valid, weight=w_valid)
    test_dm = xgb.DMatrix(X_test, label=y_test, weight=w_test)

    active_params = dict(params) if params is not None else {
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
    if random_seed is not None:
        active_params.setdefault("seed", int(random_seed))

    evals = [(train_dm, "train"), (valid_dm, "valid")]
    model = xgb.train(
        params=active_params,
        dtrain=train_dm,
        num_boost_round=250,
        evals=evals,
        early_stopping_rounds=35,
        verbose_eval=False,
    )

    preds_train = np.clip(model.predict(train_dm), 0, None)
    preds_valid = np.clip(model.predict(valid_dm), 0, None)
    preds_test = np.clip(model.predict(test_dm), 0, None)

    importance = model.get_score(importance_type="gain")
    top_importance = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
    logger.info("Top features (gain):")
    for name, score in top_importance:
        logger.info("  %s: %.4f", name, score)

    return model, preds_train, preds_valid, preds_test, (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        meta_train,
        meta_valid,
        meta_test,
        X_valid_raw,
        X_test_raw,
        encoders,
    )


def time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    meta: pd.DataFrame | None = None,
    n_splits: int = 3,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 40,
) -> list[dict[str, float]]:
    """Time-series CV for champion model."""
    if meta is None:
        if "week_start_ts" not in X.columns or "DrugId" not in X.columns:
            raise KeyError("time_series_cv requires week_start_ts and DrugId in X or a meta DataFrame.")
        order = np.argsort(X["week_start_ts"].values)
        X_ordered = X.iloc[order].reset_index(drop=True)
        y_ordered = y.iloc[order].reset_index(drop=True)
        meta_ordered = X_ordered[["week_start_ts", "DrugId"]].copy()
    else:
        meta = ensure_week_start_ts(meta)
        order = np.argsort(meta["week_start_ts"].values)
        X_ordered = X.iloc[order].reset_index(drop=True)
        y_ordered = y.iloc[order].reset_index(drop=True)
        meta_ordered = meta.iloc[order].reset_index(drop=True)
        if "DrugId" not in meta_ordered.columns:
            raise KeyError("time_series_cv requires DrugId in meta.")
    week_vals = meta_ordered["week_start_ts"].to_numpy()
    weeks = np.sort(np.unique(week_vals))
    if len(weeks) <= n_splits:
        raise ValueError("Not enough unique weeks for time-series CV.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_week_idx, valid_week_idx) in enumerate(tscv.split(weeks), start=1):
        train_weeks = weeks[train_week_idx]
        valid_weeks = weeks[valid_week_idx]
        train_mask = np.isin(week_vals, train_weeks)
        valid_mask = np.isin(week_vals, valid_weeks)

        X_train_raw, X_valid_raw = X_ordered.loc[train_mask], X_ordered.loc[valid_mask]
        y_train_raw, y_valid_raw = y_ordered.loc[train_mask], y_ordered.loc[valid_mask]
        w_all = compute_sample_weights(meta_ordered, y_ordered, train_mask=train_mask)
        w_train, w_valid = w_all[train_mask], w_all[valid_mask]

        X_train_enc, X_valid_enc, _ = encode_categoricals(X_train_raw, X_valid_raw)

        train_dm = xgb.DMatrix(X_train_enc, label=y_train_raw, weight=w_train)
        valid_dm = xgb.DMatrix(X_valid_enc, label=y_valid_raw, weight=w_valid)
        evals = [(train_dm, "train"), (valid_dm, "valid")]

        booster = xgb.train(
            params=params,
            dtrain=train_dm,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        preds = np.clip(booster.predict(valid_dm), 0, None)
        metrics = evaluate_predictions(y_valid_raw, preds, label=f"cv_fold{fold}")
        fold_metrics.append(metrics)
        logger.info("CV Fold %d: RMSE=%.3f | MAE=%.3f", fold, metrics["rmse"], metrics["mae"])

    return fold_metrics


def main(
    settings: Settings | None = None,
    data: pd.DataFrame | None = None,
    run_tag: str | None = None,
    run_dir: Path | None = None,
    run_walk_forward: bool | None = None,
) -> dict[str, float]:
    """Main training orchestrator."""
    setup_logging()
    active_settings = settings or get_settings()
    set_global_seed(active_settings.random_seed)
    active_run_walk_forward = (
        run_walk_forward if run_walk_forward is not None else active_settings.run_walk_forward
    )

    artifacts_dir = active_settings.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = active_settings.data_path() / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    active_run_dir = run_dir or create_run_dir(artifacts_dir, run_tag=run_tag)
    run_metadata = {
        "run_id": active_run_dir.name,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "seed": active_settings.random_seed,
    }
    write_json(active_run_dir / "run.json", run_metadata)
    write_json(active_run_dir / "config.json", safe_settings_snapshot(active_settings))

    df = data if data is not None else build_dataset(settings=active_settings)
    weekly_features_path = processed_dir / "weekly_features.csv"

    X, y, meta = prepare_features(df)
    diagnostics_dir = active_run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    audit_df = pd.concat([meta, X, y.rename("UPW")], axis=1)
    audit_report = audit_feature_leakage(
        audit_df, output_path=diagnostics_dir / "feature_leakage_audit.csv"
    )
    if not audit_report.empty and audit_report["mismatch_count"].sum() > 0:
        logger.warning("Leakage audit found mismatches; see diagnostics/feature_leakage_audit.csv")

    model_params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": active_settings.xgb_eta,
        "max_depth": active_settings.xgb_max_depth,
        "subsample": active_settings.xgb_subsample,
        "colsample_bytree": active_settings.xgb_colsample_bytree,
        "min_child_weight": active_settings.xgb_min_child_weight,
        "gamma": active_settings.xgb_gamma,
        "reg_lambda": active_settings.xgb_reg_lambda,
        "reg_alpha": active_settings.xgb_reg_alpha,
        "seed": int(active_settings.random_seed),
    }

    model, preds_train, preds_valid, preds_test, splits = train_xgb_model(
        X, y, meta, params=model_params, random_seed=active_settings.random_seed
    )
    (
        X_train_enc, X_valid_enc, X_test_enc,
        y_train, y_valid, y_test,
        meta_train, meta_valid, meta_test,
        X_valid_raw, X_test_raw, _encoders,
    ) = splits

    logger.info("Split summary:")
    logger.info("  Train weeks: %s | Drugs=%d", iso_range(meta_train['week_start']), meta_train['DrugId'].nunique())
    logger.info("  Valid weeks: %s | Drugs=%d", iso_range(meta_valid['week_start']), meta_valid['DrugId'].nunique())
    logger.info("  Test weeks:  %s | Drugs=%d", iso_range(meta_test['week_start']), meta_test['DrugId'].nunique())

    train_metrics = evaluate_predictions(y_train, preds_train, label="train")
    valid_metrics = evaluate_predictions(y_valid, preds_valid, label="valid")
    test_metrics = evaluate_predictions(y_test, preds_test, label="test")

    scales = calibrate_per_drug_scale(y_valid.to_numpy(), preds_valid, meta_valid["DrugId"].to_numpy())
    preds_valid_cal = apply_per_drug_scale(preds_valid, meta_valid["DrugId"].to_numpy(), scales)
    preds_test_cal = apply_per_drug_scale(preds_test, meta_test["DrugId"].to_numpy(), scales)

    if ENABLE_SPIKE_POLICY:
        preds_valid_cal = apply_spike_policy(preds_valid_cal, meta_valid, y_true=y_valid.to_numpy())
        preds_test_cal = apply_spike_policy(preds_test_cal, meta_test, y_true=None)

    abs_err = np.abs(preds_valid_cal - y_valid.to_numpy())
    error_df = pd.DataFrame({
        "DrugId": meta_valid["DrugId"].to_numpy(),
        "abs_err": abs_err,
        "y_true": y_valid.to_numpy(),
    })
    per_drug = error_df.groupby("DrugId", as_index=False).agg(
        mae=("abs_err", "mean"), mean_y=("y_true", "mean")
    )
    eps = 1e-6
    per_drug["error_rate"] = per_drug["mae"] / (per_drug["mean_y"].abs() + eps)
    per_drug["error_rate"] = per_drug["error_rate"].clip(lower=0.0, upper=1.0)
    global_error_rate = float(abs_err.mean() / (np.abs(y_valid).mean() + eps))
    global_error_rate = float(np.clip(global_error_rate, 0.0, 1.0))
    error_rate_series = meta_valid["DrugId"].map(
        per_drug.set_index("DrugId")["error_rate"]
    ).fillna(global_error_rate).clip(lower=0.0, upper=1.0)

    pred_low, pred_mid, pred_high = build_error_band(preds_valid_cal, error_rate_series.to_numpy())
    valid_cal_metrics = evaluate_predictions(y_valid, preds_valid_cal, label="valid_calibrated")
    test_cal_metrics = evaluate_predictions(y_test, preds_test_cal, label="test_calibrated")

    print_evaluation(train_metrics, valid_metrics, test_metrics)
    logger.info("Valid metrics (per-drug scale): RMSE=%.3f | MAE=%.3f | R2=%.3f",
                valid_cal_metrics['rmse'], valid_cal_metrics['mae'], valid_cal_metrics['r2'])
    logger.info("Test metrics (per-drug scale): RMSE=%.3f | MAE=%.3f | R2=%.3f",
                test_cal_metrics['rmse'], test_cal_metrics['mae'], test_cal_metrics['r2'])

    logger.info("Time-series cross-validation (rolling splits)")
    cv_metrics = time_series_cv(X, y, params=model_params, meta=meta, n_splits=active_settings.cv_n_splits)
    avg_rmse = np.mean([m["rmse"] for m in cv_metrics])
    avg_mae = np.mean([m["mae"] for m in cv_metrics])
    logger.info("Average CV RMSE: %.3f | Average CV MAE: %.3f", avg_rmse, avg_mae)

    pred_dir = active_run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    wf_metrics: dict[str, float] = {}
    if active_run_walk_forward:
        run_wf_path = pred_dir / "upw_walkforward_predictions.csv"
        logger.info("Walk-forward (rolling-origin) with rolling MAE-based bands")
        wf_df, wf_metrics = walk_forward_forecast(
            X, y, meta, params=model_params, output_path=str(run_wf_path)
        )
        fold1_path = diagnostics_dir / "walkforward_fold1_drugs.csv"
        summarize_walkforward_fold1(wf_df, meta, output_path=fold1_path)
        copy_if_exists(fold1_path, processed_dir / fold1_path.name)
        copy_if_exists(run_wf_path, processed_dir / run_wf_path.name)
        logger.info("Walk-forward: RMSE=%.3f | MAE=%.3f", wf_metrics['rmse'], wf_metrics['mae'])
    else:
        logger.info("Walk-forward backtest disabled.")

    valid_preview = meta_valid.reset_index(drop=True).copy()
    features_preview = X_valid_raw.reset_index(drop=True).copy()
    valid_preview = pd.concat([valid_preview, features_preview], axis=1)
    valid_preview["y_true"] = y_valid.values
    valid_preview["y_pred"] = preds_valid
    valid_preview["y_pred_cal"] = preds_valid_cal

    q_map = compute_residual_quantiles(y_valid.to_numpy(), preds_valid_cal, meta_valid["DrugId"].to_numpy())
    q80, q90 = apply_residual_quantiles(preds_valid_cal, meta_valid["DrugId"].to_numpy(), q_map)
    valid_preview["y_pred_q80"] = q80
    valid_preview["y_pred_q90"] = q90
    valid_preview["y_pred_low"] = pred_low
    valid_preview["y_pred_mid"] = pred_mid
    valid_preview["y_pred_high"] = pred_high
    valid_preview["error_rate"] = error_rate_series.to_numpy()
    valid_preview["y_pred_round"] = np.round(valid_preview["y_pred"]).astype(int)
    valid_preview["y_pred_cal_round"] = np.round(valid_preview["y_pred_cal"]).astype(int)

    low_value, mid_value, high_value = compute_decile_values(
        valid_preview["y_pred_low"].to_numpy(), valid_preview["y_pred_high"].to_numpy()
    )
    valid_preview["low_value"] = low_value
    valid_preview["mid_value"] = mid_value
    valid_preview["high_value"] = high_value
    valid_preview["residual"] = valid_preview["y_true"] - valid_preview["y_pred"]
    valid_preview["residual_cal"] = valid_preview["y_true"] - valid_preview["y_pred_cal"]

    run_valid_path = pred_dir / "upw_valid_predictions.csv"
    write_dual_csv(valid_preview, run_valid_path, processed_dir / run_valid_path.name)

    run_drift_path = diagnostics_dir / "upw_fold_drift.csv"
    monitor_drift(X_train_enc, X_valid_enc, output_path=str(run_drift_path))
    copy_if_exists(run_drift_path, processed_dir / run_drift_path.name)
    monitor_rolling_rmse(y_valid.reset_index(drop=True), preds_valid_cal)

    run_model_path = active_run_dir / "models" / "upw_xgb_model.json"
    run_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(run_model_path)
    latest_model_path = artifacts_dir / "models" / "upw_xgb_model.json"
    copy_if_exists(run_model_path, latest_model_path)
    logger.info("Model saved to %s", run_model_path)

    meta = ensure_week_start_ts(meta)
    order = np.argsort(meta["week_start_ts"].values)
    y_ordered = y.iloc[order].reset_index(drop=True)
    meta_ordered = meta.iloc[order].reset_index(drop=True)
    train_weeks, valid_weeks, test_weeks = three_year_split(meta_ordered)
    train_mask = meta_ordered["week_start_ts"].isin(train_weeks).to_numpy()
    valid_mask = meta_ordered["week_start_ts"].isin(valid_weeks).to_numpy()
    test_mask = meta_ordered["week_start_ts"].isin(test_weeks).to_numpy()

    baseline_preds = baseline_predictions(y_ordered, meta_ordered)
    baseline_metrics = {
        "naive": {
            "train": evaluate_with_mask(y_ordered, baseline_preds["pred_naive"], train_mask, "naive_train"),
            "valid": evaluate_with_mask(y_ordered, baseline_preds["pred_naive"], valid_mask, "naive_valid"),
            "test": evaluate_with_mask(y_ordered, baseline_preds["pred_naive"], test_mask, "naive_test"),
        },
        "moving_avg": {
            "train": evaluate_with_mask(y_ordered, baseline_preds["pred_moving_avg"], train_mask, "ma_train"),
            "valid": evaluate_with_mask(y_ordered, baseline_preds["pred_moving_avg"], valid_mask, "ma_valid"),
            "test": evaluate_with_mask(y_ordered, baseline_preds["pred_moving_avg"], test_mask, "ma_test"),
        },
        "seasonal_naive": {
            "train": evaluate_with_mask(y_ordered, baseline_preds["pred_seasonal_naive"], train_mask, "seasonal_naive_train"),
            "valid": evaluate_with_mask(y_ordered, baseline_preds["pred_seasonal_naive"], valid_mask, "seasonal_naive_valid"),
            "test": evaluate_with_mask(y_ordered, baseline_preds["pred_seasonal_naive"], test_mask, "seasonal_naive_test"),
        },
    }

    shuffle_metrics = shuffle_target_test(X, y, meta, random_seed=active_settings.random_seed)
    logger.info("Shuffled-target test: RMSE=%.3f | R2=%.3f | MAE=%.3f",
                shuffle_metrics['rmse'], shuffle_metrics['r2'], shuffle_metrics['mae'])

    data_manifest = build_data_manifest(df, meta_train, meta_valid, meta_test, weekly_features_path)
    train_params = {
        "xgb_params": model_params,
        "split_weeks": {"train": "years_1_2", "valid": "year3_h1", "test": "year3_h2"},
        "num_boost_round": 250,
        "early_stopping_rounds": 35,
        "weights": {
            "time_decay_half_life_weeks": 26,
            "volume_weight": "sqrt(train_total/median)",
            "censored_weight": 0.5,
        },
        "walk_forward": {
            "enabled": bool(active_run_walk_forward),
            "min_train_weeks": 30,
            "error_window": 8,
        },
    }
    metrics_payload = {
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
        "valid_calibrated": valid_cal_metrics,
        "test_calibrated": test_cal_metrics,
        "cv": {"folds": cv_metrics, "avg_rmse": avg_rmse, "avg_mae": avg_mae},
        "walk_forward": wf_metrics,
        "baselines": baseline_metrics,
        "shuffle_test": shuffle_metrics,
    }

    write_json(active_run_dir / "data_manifest.json", data_manifest)
    write_json(active_run_dir / "train_params.json", train_params)
    write_json(active_run_dir / "metrics.json", metrics_payload)
    metrics_dir = artifacts_dir / "metrics" / active_run_dir.name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "metrics.json", metrics_payload)
    return metrics_payload


if __name__ == "__main__":
    main()
