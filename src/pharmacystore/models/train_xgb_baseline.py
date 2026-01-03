"""Train an XGBoost model to forecast weekly UPW using all engineered features."""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

from pharmacystore.config import Settings, get_settings
from pharmacystore.pipeline import run_pipeline
from pharmacystore.run_utils import compute_file_hash, create_run_dir, set_global_seed, write_json


def build_dataset(settings: Settings | None = None) -> pd.DataFrame:
    """Build the full weekly feature set via the existing pipeline."""
    weekly_df = run_pipeline(settings=settings)
    return weekly_df


def _safe_settings_snapshot(settings: Settings) -> dict:
    snapshot = settings.model_dump()
    password = snapshot.pop("sql_password", None)
    snapshot["sql_password_set"] = bool(password)
    return snapshot


def _iso_range(series: pd.Series) -> dict[str, str | None]:
    if series is None:
        return {"min": None, "max": None}
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "max": None}
    return {
        "min": values.min().date().isoformat(),
        "max": values.max().date().isoformat(),
    }


def _write_dual_csv(df: pd.DataFrame, run_path: Path, latest_path: Path) -> None:
    run_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_path, index=False)
    if run_path != latest_path:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(latest_path, index=False)


def _copy_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _build_data_manifest(
    df: pd.DataFrame,
    meta_train: pd.DataFrame,
    meta_valid: pd.DataFrame,
    meta_test: pd.DataFrame,
    weekly_features_path: Path,
) -> dict:
    manifest = {
        "weekly_rows": int(len(df)),
        "unique_drugs": int(df["DrugId"].nunique()) if "DrugId" in df.columns else None,
        "unique_weeks": int(df["week_start"].nunique()) if "week_start" in df.columns else None,
        "week_start_range": _iso_range(df.get("week_start")),
        "train_rows": int(len(meta_train)),
        "valid_rows": int(len(meta_valid)),
        "test_rows": int(len(meta_test)),
        "train_week_range": _iso_range(meta_train.get("week_start")),
        "valid_week_range": _iso_range(meta_valid.get("week_start")),
        "test_week_range": _iso_range(meta_test.get("week_start")),
    }
    if weekly_features_path.exists():
        manifest["weekly_features_hash_md5"] = compute_file_hash(weekly_features_path)
    return manifest


def _split_weeks(
    meta: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def prepare_features(df: pd.DataFrame):
    """Encode features for XGBoost; return X, y, meta, and encoders."""
    data = df.copy()
    # Convert date to numeric timestamp (seconds since epoch)
    data["week_start_ts"] = pd.to_datetime(data["week_start"]).view("int64") // 10**9

    meta_cols = [col for col in ["week_start", "week_start_ts", "DrugId"] if col in data.columns]
    meta = data[meta_cols].copy()

    # Target and features
    y = data["UPW"].astype(float)
    X = data.drop(columns=["UPW", "week_start"])

    # Label-encode categorical columns
    encoders: dict[str, LabelEncoder] = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna("NA"))
        encoders[col] = le

    # Fill any remaining NaNs (e.g., from rolling std at early points)
    X = X.fillna(0)
    return X, y, meta, encoders


def train_xgb_model(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    params: dict | None = None,
    random_seed: int | None = None,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
):
    """Train/validate an XGBoost regressor using a time-based split."""
    # Time-based ordering to avoid leakage
    order = np.argsort(X["week_start_ts"].values)
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)
    meta_ordered = meta.iloc[order].reset_index(drop=True)

    train_weeks, valid_weeks, test_weeks = _split_weeks(
        meta_ordered, train_frac=train_frac, valid_frac=valid_frac
    )
    train_mask = meta_ordered["week_start_ts"].isin(train_weeks).to_numpy()
    valid_mask = meta_ordered["week_start_ts"].isin(valid_weeks).to_numpy()
    test_mask = meta_ordered["week_start_ts"].isin(test_weeks).to_numpy()

    X_train = X_ordered.loc[train_mask].reset_index(drop=True)
    X_valid = X_ordered.loc[valid_mask].reset_index(drop=True)
    X_test = X_ordered.loc[test_mask].reset_index(drop=True)
    y_train = y_ordered.loc[train_mask].reset_index(drop=True)
    y_valid = y_ordered.loc[valid_mask].reset_index(drop=True)
    y_test = y_ordered.loc[test_mask].reset_index(drop=True)
    meta_train = meta_ordered.loc[train_mask].reset_index(drop=True)
    meta_valid = meta_ordered.loc[valid_mask].reset_index(drop=True)
    meta_test = meta_ordered.loc[test_mask].reset_index(drop=True)

    train_dm = xgb.DMatrix(X_train, label=y_train)
    valid_dm = xgb.DMatrix(X_valid, label=y_valid)
    test_dm = xgb.DMatrix(X_test, label=y_test)

    active_params = dict(params) if params is not None else {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
    }
    if random_seed is not None:
        active_params.setdefault("seed", int(random_seed))

    evals = [(train_dm, "train"), (valid_dm, "valid")]
    model = xgb.train(
        params=active_params,
        dtrain=train_dm,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    preds_train = model.predict(train_dm)
    preds_valid = model.predict(valid_dm)
    preds_test = model.predict(test_dm)

    # Top feature importances
    importance = model.get_score(importance_type="gain")
    top_importance = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop features (gain):")
    for name, score in top_importance:
        print(f"{name}: {score:.4f}")

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
    )


def evaluate_predictions(y_true: pd.Series, preds: np.ndarray, label: str = "") -> dict[str, float]:
    """Compute a set of regression metrics."""
    eps = 1e-6
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    medae = median_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-6))) * 100
    bias = float(np.mean(preds - y_true))
    wape = float(np.sum(np.abs(y_true - preds)) / (np.sum(np.abs(y_true)) + eps))
    smape = float(
        np.mean(2 * np.abs(y_true - preds) / (np.abs(y_true) + np.abs(preds) + eps))
    )
    metrics = {
        "label": label,
        "rmse": rmse,
        "mae": mae,
        "median_ae": medae,
        "r2": r2,
        "mape_pct": mape,
        "bias": bias,
        "wape": wape,
        "smape": smape,
    }
    return metrics


def print_evaluation(
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
    test_metrics: dict[str, float] | None = None,
) -> None:
    """Pretty-print metrics for train/validation/test splits."""
    def _fmt(m: dict[str, float]) -> str:
        return (
            f"RMSE={m['rmse']:.3f} | MAE={m['mae']:.3f} | MedAE={m['median_ae']:.3f} | "
            f"R2={m['r2']:.3f} | WAPE={m['wape']:.3f} | sMAPE={m['smape']:.3f} | "
            f"MAPE={m['mape_pct']:.2f}% | Bias={m['bias']:.3f}"
        )

    print("\n=== Evaluation ===")
    print(f"Train: {_fmt(train_metrics)}")
    print(f"Valid: {_fmt(valid_metrics)}")
    if test_metrics is not None:
        print(f"Test: {_fmt(test_metrics)}")


def main(
    settings: Settings | None = None,
    data: pd.DataFrame | None = None,
    run_tag: str | None = None,
    run_dir: Path | None = None,
) -> dict[str, float]:
    active_settings = settings or get_settings()
    set_global_seed(active_settings.random_seed)

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
    write_json(active_run_dir / "config.json", _safe_settings_snapshot(active_settings))

    df = data if data is not None else build_dataset(settings=active_settings)
    weekly_features_path = processed_dir / "weekly_features.csv"

    X, y, meta, _encoders = prepare_features(df)
    model_params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "seed": int(active_settings.random_seed),
    }
    model, preds_train, preds_valid, preds_test, splits = train_xgb_model(
        X,
        y,
        meta,
        params=model_params,
        random_seed=active_settings.random_seed,
        train_frac=0.7,
        valid_frac=0.15,
    )
    X_train, X_valid, X_test, y_train, y_valid, y_test, meta_train, meta_valid, meta_test = splits

    train_metrics = evaluate_predictions(y_train, preds_train, label="train")
    valid_metrics = evaluate_predictions(y_valid, preds_valid, label="valid")
    test_metrics = evaluate_predictions(y_test, preds_test, label="test")
    print_evaluation(train_metrics, valid_metrics, test_metrics)

    pred_dir = active_run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Export a small validation preview with predictions and residuals
    valid_preview = X_valid.copy()
    valid_preview["y_true"] = y_valid.values
    valid_preview["y_pred"] = preds_valid
    valid_preview["residual"] = valid_preview["y_true"] - valid_preview["y_pred"]
    run_valid_path = pred_dir / "upw_valid_predictions.csv"
    _write_dual_csv(valid_preview, run_valid_path, processed_dir / run_valid_path.name)

    # Save model and minimal artifacts for reuse
    run_model_path = active_run_dir / "models" / "upw_xgb_baseline.json"
    run_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(run_model_path)
    latest_model_path = artifacts_dir / "models" / "upw_xgb_baseline.json"
    _copy_if_exists(run_model_path, latest_model_path)
    print(f"\nModel saved to {run_model_path}")
    print(f"Train size: {len(X_train)}, Validation size: {len(X_valid)}, Test size: {len(X_test)}")

    data_manifest = _build_data_manifest(df, meta_train, meta_valid, meta_test, weekly_features_path)
    train_params = {
        "xgb_params": model_params,
        "split_weeks": {"train": 0.7, "valid": 0.15, "test": 0.15},
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    }
    metrics_payload = {"train": train_metrics, "valid": valid_metrics, "test": test_metrics}

    write_json(active_run_dir / "data_manifest.json", data_manifest)
    write_json(active_run_dir / "train_params.json", train_params)
    write_json(active_run_dir / "metrics.json", metrics_payload)
    return metrics_payload


if __name__ == "__main__":
    main()
