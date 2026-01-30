from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

from pharmacystore.config import get_settings, Settings
from pharmacystore.pipeline import run_pipeline
from pharmacystore.models.train_xgb import (
    prepare_features,
    encode_categoricals,
    apply_label_encoders,
    _split_weeks,
    _three_year_split,
    compute_sample_weights,
    _select_baseline_features,
    apply_shock_guard,
)


@dataclass
class AblationConfig:
    name: str
    split_mode: str  # "simple" | "year3"
    use_new_features: bool
    use_weights: bool
    use_two_stage: bool
    use_shock_guard: bool
    target_mode: str  # "raw" | "log1p" | "log1p_spike"


def _load_weekly_features(settings: Settings) -> pd.DataFrame:
    processed = settings.data_path() / "processed" / "weekly_features.csv"
    if processed.exists():
        df = pd.read_csv(processed, parse_dates=["week_start"])
        return df
    return run_pipeline(settings=settings)


def _split_masks_with_info(
    meta: pd.DataFrame, split_mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if split_mode == "year3":
        train_weeks, valid_weeks, test_weeks = _three_year_split(meta, min_valid_weeks=4, min_test_weeks=4)
    else:
        train_weeks, valid_weeks, test_weeks = _split_weeks(meta, train_frac=0.7, valid_frac=0.15)
    train_mask = meta["week_start_ts"].isin(train_weeks).to_numpy()
    valid_mask = meta["week_start_ts"].isin(valid_weeks).to_numpy()
    test_mask = meta["week_start_ts"].isin(test_weeks).to_numpy()

    split_info = {
        "train_weeks": int(len(set(train_weeks))),
        "valid_weeks": int(len(set(valid_weeks))),
        "test_weeks": int(len(set(test_weeks))),
        "train_range": str(meta.loc[train_mask, "week_start"].min()) + " -> " + str(meta.loc[train_mask, "week_start"].max()),
        "valid_range": str(meta.loc[valid_mask, "week_start"].min()) + " -> " + str(meta.loc[valid_mask, "week_start"].max()),
        "test_range": str(meta.loc[test_mask, "week_start"].min()) + " -> " + str(meta.loc[test_mask, "week_start"].max()),
        "split_applied": split_mode,
    }

    if split_mode == "year3":
        weeks_df = meta[["week_start_ts", "week_start"]].drop_duplicates().sort_values("week_start_ts")
        week_dates = pd.to_datetime(weeks_df["week_start"])
        base_year = int(week_dates.min().year)
        weeks_df["rel_year"] = week_dates.dt.year - base_year + 1
        weeks_df["month"] = week_dates.dt.month
        valid_set = set(valid_weeks)
        test_set = set(test_weeks)
        valid_is_year3_h1 = weeks_df[
            weeks_df["week_start_ts"].isin(valid_set)
        ].eval("rel_year == 3 and month <= 6").all()
        test_is_year3_h2 = weeks_df[
            weeks_df["week_start_ts"].isin(test_set)
        ].eval("rel_year == 3 and month >= 7").all()
        if not (valid_is_year3_h1 and test_is_year3_h2):
            split_info["split_applied"] = "fallback_simple"
    return train_mask, valid_mask, test_mask, split_info


def _feature_drop_list() -> list[str]:
    return [
        "diff_rate",
        "Z_score",
        "trend_short_vs_long",
        "trend_slope_8w",
        "trend_accel",
        "trend_slope_8w_norm",
        "pct_trend_8w",
        "seasonality_strength_8w",
        "regime_prob_8w",
        "week_sin",
        "week_cos",
        "spike_score_8",
    ]


def _prepare_ablation_features(
    X: pd.DataFrame, use_new_features: bool
) -> pd.DataFrame:
    if use_new_features:
        return X.copy()
    drop_cols = [c for c in _feature_drop_list() if c in X.columns]
    return X.drop(columns=drop_cols, errors="ignore")


def _target_transform(
    y: pd.Series, spike_flag: np.ndarray, mode: str, alpha: float
) -> np.ndarray:
    if mode == "raw":
        return y.to_numpy()
    if mode == "log1p":
        return np.log1p(y.to_numpy())
    if mode == "log1p_spike":
        return np.log1p(y.to_numpy()) + alpha * spike_flag
    raise ValueError(f"Unknown target_mode: {mode}")


def _inverse_target(
    pred: np.ndarray, spike_flag: np.ndarray, mode: str, alpha: float
) -> np.ndarray:
    if mode == "raw":
        return np.clip(pred, 0, None)
    if mode == "log1p":
        return np.clip(np.expm1(pred), 0, None)
    if mode == "log1p_spike":
        return np.clip(np.expm1(pred - alpha * spike_flag), 0, None)
    raise ValueError(f"Unknown target_mode: {mode}")


def _fit_baseline(
    base_features: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    mode: str,
    spike_flag: np.ndarray,
    alpha: float,
) -> tuple[Ridge, np.ndarray]:
    y_target = _target_transform(pd.Series(y_train), spike_flag, mode, alpha)
    model = Ridge(alpha=1.0)
    model.fit(base_features, y_target, sample_weight=w_train)
    base_pred = model.predict(base_features)
    return model, base_pred


def _train_predict(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    cfg: AblationConfig,
    xgb_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha_spike = 0.35
    spike_flag = X.get("spike_flag", pd.Series(0, index=X.index)).fillna(0).to_numpy()
    X_used = _prepare_ablation_features(X, cfg.use_new_features)
    train_mask, valid_mask, test_mask, _ = _split_masks_with_info(meta, cfg.split_mode)
    weights = (
        compute_sample_weights(meta, y, train_mask=train_mask)
        if cfg.use_weights
        else np.ones(len(meta))
    )

    X_train_raw = X_used.loc[train_mask].reset_index(drop=True)
    X_valid_raw = X_used.loc[valid_mask].reset_index(drop=True)
    X_test_raw = X_used.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_valid = y.loc[valid_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    w_train = weights[train_mask]
    w_valid = weights[valid_mask]
    w_test = weights[test_mask]

    # Baseline features are derived from the same feature set to avoid leakage in ablation
    base_features = _select_baseline_features(X_used)
    base_train = base_features.loc[train_mask].reset_index(drop=True)
    base_valid = base_features.loc[valid_mask].reset_index(drop=True)
    base_test = base_features.loc[test_mask].reset_index(drop=True)

    base_model = None
    base_pred_train = None
    base_pred_valid = None
    base_pred_test = None

    if cfg.use_two_stage or cfg.use_shock_guard:
        base_model, base_pred_train = _fit_baseline(
            base_train,
            y_train.to_numpy(),
            w_train,
            "log1p",
            spike_flag[train_mask],
            alpha_spike,
        )
        base_pred_valid = base_model.predict(base_valid)
        base_pred_test = base_model.predict(base_test)

    if cfg.use_two_stage:
        y_target = _target_transform(y, spike_flag, cfg.target_mode, alpha_spike)
        y_target_train = y_target[train_mask]
        y_target_valid = y_target[valid_mask]
        y_target_test = y_target[test_mask]

        X_train_stage = X_train_raw.copy()
        X_valid_stage = X_valid_raw.copy()
        X_test_stage = X_test_raw.copy()
        X_train_stage["baseline_pred_log"] = base_pred_train
        X_valid_stage["baseline_pred_log"] = base_pred_valid
        X_test_stage["baseline_pred_log"] = base_pred_test

        residual_train = y_target_train - base_pred_train
        residual_valid = y_target_valid - base_pred_valid

        X_train_enc, X_valid_enc, encoders = encode_categoricals(X_train_stage, X_valid_stage)
        X_test_enc = apply_label_encoders(X_test_stage, encoders)

        train_dm = xgb.DMatrix(X_train_enc, label=residual_train, weight=w_train)
        valid_dm = xgb.DMatrix(X_valid_enc, label=residual_valid, weight=w_valid)
        test_dm = xgb.DMatrix(X_test_enc, weight=w_test)

        booster = xgb.train(
            params=xgb_params,
            dtrain=train_dm,
            num_boost_round=200,
            evals=[(train_dm, "train"), (valid_dm, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        resid_valid = booster.predict(valid_dm)
        resid_test = booster.predict(test_dm)
        pred_valid = _inverse_target(base_pred_valid + resid_valid, spike_flag[valid_mask], cfg.target_mode, alpha_spike)
        pred_test = _inverse_target(base_pred_test + resid_test, spike_flag[test_mask], cfg.target_mode, alpha_spike)
    else:
        y_target = _target_transform(y, spike_flag, cfg.target_mode, alpha_spike)
        y_target_train = y_target[train_mask]
        y_target_valid = y_target[valid_mask]
        y_target_test = y_target[test_mask]

        X_train_enc, X_valid_enc, encoders = encode_categoricals(X_train_raw, X_valid_raw)
        X_test_enc = apply_label_encoders(X_test_raw, encoders)

        train_dm = xgb.DMatrix(X_train_enc, label=y_target_train, weight=w_train)
        valid_dm = xgb.DMatrix(X_valid_enc, label=y_target_valid, weight=w_valid)
        test_dm = xgb.DMatrix(X_test_enc, weight=w_test)

        booster = xgb.train(
            params=xgb_params,
            dtrain=train_dm,
            num_boost_round=200,
            evals=[(train_dm, "train"), (valid_dm, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        pred_valid = _inverse_target(booster.predict(valid_dm), spike_flag[valid_mask], cfg.target_mode, alpha_spike)
        pred_test = _inverse_target(booster.predict(test_dm), spike_flag[test_mask], cfg.target_mode, alpha_spike)

    if cfg.use_shock_guard:
        base_valid_sales = np.clip(np.expm1(base_pred_valid), 0, None)
        base_test_sales = np.clip(np.expm1(base_pred_test), 0, None)
        pred_valid = apply_shock_guard(pred_valid, base_valid_sales, X_valid_raw)
        pred_test = apply_shock_guard(pred_test, base_test_sales, X_test_raw)

    valid_df = pd.DataFrame(
        {
            "DrugId": meta.loc[valid_mask, "DrugId"].to_numpy(),
            "week_start": meta.loc[valid_mask, "week_start"].to_numpy(),
            "y_true": y_valid.to_numpy(),
            "y_pred": pred_valid,
        }
    )
    test_df = pd.DataFrame(
        {
            "DrugId": meta.loc[test_mask, "DrugId"].to_numpy(),
            "week_start": meta.loc[test_mask, "week_start"].to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_pred": pred_test,
        }
    )
    return valid_df, test_df


def _select_top_drugs(meta: pd.DataFrame, y: pd.Series, train_mask: np.ndarray, top_n: int = 50) -> list:
    train_meta = meta.loc[train_mask].reset_index(drop=True)
    train_y = y.loc[train_mask].reset_index(drop=True)
    totals = train_y.groupby(train_meta["DrugId"]).sum().sort_values(ascending=False)
    return totals.head(top_n).index.tolist()


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-6
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def _wwape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-6
    if len(y_true) == 0:
        return 0.0
    high_cut = np.quantile(np.abs(y_true), 0.75)
    weights = np.where(np.abs(y_true) >= high_cut, 2.0, 1.0)
    return float(
        np.sum(weights * np.abs(y_true - y_pred)) / (np.sum(weights * np.abs(y_true)) + eps)
    )


def _under_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    under = y_pred < y_true
    under_rate = float(np.mean(under)) if len(y_true) else 0.0
    under_share = float(np.sum((y_true - y_pred) * under) / (np.sum(np.abs(y_true)) + eps))
    return under_rate, under_share


def _worst_week_error(df: pd.DataFrame) -> float:
    eps = 1e-6
    weekly = (
        df.groupby("week_start")
        .apply(lambda g: np.sum(np.abs(g["y_true"] - g["y_pred"])) / (np.sum(np.abs(g["y_true"])) + eps))
    )
    if weekly.empty:
        return 0.0
    return float(np.quantile(weekly, 0.99))


def _evaluate_subset(df: pd.DataFrame, top_drugs: Iterable) -> dict:
    subset = df[df["DrugId"].isin(set(top_drugs))].copy()
    y_true = subset["y_true"].to_numpy()
    y_pred = subset["y_pred"].to_numpy()
    wape = _wape(y_true, y_pred)
    wwape = _wwape(y_true, y_pred)
    under_rate, under_share = _under_metrics(y_true, y_pred)
    worst_week = _worst_week_error(subset)
    return {
        "wape": wape,
        "wwape": wwape,
        "under_rate": under_rate,
        "under_share": under_share,
        "worst_1pct_week": worst_week,
    }


def run_ablation(settings: Settings, output_path: Path, top_n: int) -> pd.DataFrame:
    df = _load_weekly_features(settings)
    X, y, meta = prepare_features(df)

    base_xgb_params = {
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

    configs = [
        AblationConfig(
            name="base",
            split_mode="simple",
            use_new_features=False,
            use_weights=False,
            use_two_stage=False,
            use_shock_guard=False,
            target_mode="raw",
        ),
        AblationConfig(
            name="split_only",
            split_mode="year3",
            use_new_features=False,
            use_weights=False,
            use_two_stage=False,
            use_shock_guard=False,
            target_mode="raw",
        ),
        AblationConfig(
            name="features_only",
            split_mode="simple",
            use_new_features=True,
            use_weights=False,
            use_two_stage=False,
            use_shock_guard=False,
            target_mode="raw",
        ),
        AblationConfig(
            name="weights_only",
            split_mode="simple",
            use_new_features=False,
            use_weights=True,
            use_two_stage=False,
            use_shock_guard=False,
            target_mode="raw",
        ),
        AblationConfig(
            name="two_stage_only",
            split_mode="simple",
            use_new_features=False,
            use_weights=False,
            use_two_stage=True,
            use_shock_guard=False,
            target_mode="log1p",
        ),
        AblationConfig(
            name="shock_guard_only",
            split_mode="simple",
            use_new_features=False,
            use_weights=False,
            use_two_stage=False,
            use_shock_guard=True,
            target_mode="raw",
        ),
        AblationConfig(
            name="all_current",
            split_mode="year3",
            use_new_features=True,
            use_weights=True,
            use_two_stage=True,
            use_shock_guard=True,
            target_mode="log1p_spike",
        ),
    ]

    rows = []
    for cfg in configs:
        train_mask, _, _, split_info = _split_masks_with_info(meta, cfg.split_mode)
        top50 = _select_top_drugs(meta, y, train_mask, top_n=top_n)
        valid_df, test_df = _train_predict(X, y, meta, cfg, base_xgb_params)

        valid_metrics = _evaluate_subset(valid_df, top50)
        test_metrics = _evaluate_subset(test_df, top50)
        rows.append(
            {
                "variant": cfg.name,
                "split": cfg.split_mode,
                "split_applied": split_info["split_applied"],
                "train_range": split_info["train_range"],
                "valid_range": split_info["valid_range"],
                "test_range": split_info["test_range"],
                "train_weeks": split_info["train_weeks"],
                "valid_weeks": split_info["valid_weeks"],
                "test_weeks": split_info["test_weeks"],
                "new_features": cfg.use_new_features,
                "weights": cfg.use_weights,
                "two_stage": cfg.use_two_stage,
                "shock_guard": cfg.use_shock_guard,
                "target_mode": cfg.target_mode,
                f"valid_wape_top{top_n}": valid_metrics["wape"],
                f"valid_wwape_top{top_n}": valid_metrics["wwape"],
                f"valid_under_rate_top{top_n}": valid_metrics["under_rate"],
                f"valid_under_share_top{top_n}": valid_metrics["under_share"],
                f"valid_worst_1pct_week_top{top_n}": valid_metrics["worst_1pct_week"],
                f"test_wape_top{top_n}": test_metrics["wape"],
                f"test_wwape_top{top_n}": test_metrics["wwape"],
                f"test_under_rate_top{top_n}": test_metrics["under_rate"],
                f"test_under_share_top{top_n}": test_metrics["under_share"],
                f"test_worst_1pct_week_top{top_n}": test_metrics["worst_1pct_week"],
            }
        )

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved ablation report to {output_path}")
    print(out_df.to_string(index=False))
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study for forecasting pipeline.")
    parser.add_argument("--output", default="artifacts/ablation_report.csv")
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()
    settings = get_settings()
    run_ablation(settings, Path(args.output), args.top_n)


if __name__ == "__main__":
    main()
