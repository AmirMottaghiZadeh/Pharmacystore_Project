from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


METRICS = ["mae", "rmse", "wape", "smape", "under_forecast_rate"]
SPLITS = [
    ("train", "Train"),
    ("valid", "Valid"),
    ("test", "Test"),
    ("walk_forward", "Walk-forward"),
    ("valid_calibrated", "Valid (calibrated)"),
    ("test_calibrated", "Test (calibrated)"),
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def _load_json(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner=False)
def _load_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _list_runs(runs_root: Path) -> list[str]:
    if not runs_root.exists():
        return []
    runs = [p.name for p in runs_root.iterdir() if p.is_dir()]
    return sorted(runs, reverse=True)


def _coalesce_column(df: pd.DataFrame, target: str, *candidates: str) -> None:
    for candidate in candidates:
        if candidate in df.columns:
            if candidate != target:
                df[target] = df[candidate]
            return
    if target not in df.columns:
        df[target] = pd.NA


def _value_segments(values: pd.Series) -> pd.Series:
    if values.dropna().empty:
        return pd.Series(["Unknown"] * len(values), index=values.index)
    q1 = values.quantile(1 / 3)
    q2 = values.quantile(2 / 3)

    def _segment(value: float | int | None) -> str:
        if pd.isna(value):
            return "Unknown"
        if value <= q1:
            return "low_value"
        if value <= q2:
            return "mid_value"
        return "high_value"

    return values.apply(_segment)


def _normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    _coalesce_column(out, "week_start", "week_start")
    _coalesce_column(out, "DrugId", "DrugId", "DrugId.1")
    _coalesce_column(out, "genericname", "genericname", "genericname.1")
    _coalesce_column(out, "saleCategory", "saleCategory")
    _coalesce_column(out, "priceCategory", "priceCategory")
    _coalesce_column(out, "classified_drug", "classified_drug")
    _coalesce_column(out, "y_true", "y_true")
    _coalesce_column(out, "y_pred", "y_pred")
    _coalesce_column(out, "y_pred_cal", "y_pred_cal")
    _coalesce_column(out, "y_pred_low", "y_pred_low")
    _coalesce_column(out, "y_pred_high", "y_pred_high")
    _coalesce_column(out, "spike_score_8", "spike_score_8")
    _coalesce_column(out, "error_rate", "error_rate")

    for col in ["y_true", "y_pred", "y_pred_cal", "y_pred_low", "y_pred_high", "spike_score_8", "error_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["DrugId", "genericname", "saleCategory", "priceCategory", "classified_drug"]:
        out[col] = out[col].fillna("Unknown").astype(str).str.strip()
        out.loc[out[col].isin({"", "nan", "None"}), col] = "Unknown"

    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out = out[out["week_start"].notna()].copy()
    out["value_segment"] = _value_segments(out["y_true"])

    if out["error_rate"].dropna().empty:
        denom = out["y_true"].replace(0, pd.NA)
        out["error_rate"] = (out["y_true"] - out["y_pred"]).abs() / denom

    return out.sort_values(["week_start", "DrugId"]).reset_index(drop=True)


def _normalize_drift(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "feature" not in out.columns:
        out = out.rename(columns={out.columns[0]: "feature"})

    for col in ["train_mean", "valid_mean", "mean_diff", "mean_pct", "train_std", "valid_std", "std_diff"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA

    out["feature"] = out["feature"].fillna("Unknown").astype(str)
    return out.sort_values("mean_pct", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def _metric(metrics: dict[str, Any], split: str, name: str) -> float | None:
    section = metrics.get(split, {})
    if not isinstance(section, dict):
        return None
    value = section.get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _baseline_metric(metrics: dict[str, Any], split: str, baseline: str, name: str) -> float | None:
    baselines = metrics.get("baselines", {})
    split_section = baselines.get(split, {}) if isinstance(baselines, dict) else {}
    baseline_section = split_section.get(baseline, {}) if isinstance(split_section, dict) else {}
    value = baseline_section.get(name) if isinstance(baseline_section, dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def _load_run_bundle(runs_root_str: str, data_root_str: str, run_id: str) -> dict[str, Any]:
    runs_root = Path(runs_root_str)
    data_root = Path(data_root_str)
    run_root = runs_root / run_id
    processed_root = data_root / "processed"

    metrics = _load_json(str(run_root / "metrics.json"))
    manifest = _load_json(str(run_root / "data_manifest.json"))
    params = _load_json(str(run_root / "train_params.json"))
    config = _load_json(str(run_root / "config.json"))
    run_meta = _load_json(str(run_root / "run.json"))

    valid_path = run_root / "predictions" / "upw_valid_predictions.csv"
    walk_path = run_root / "predictions" / "upw_walkforward_predictions.csv"
    drift_path = run_root / "diagnostics" / "upw_fold_drift.csv"

    if not valid_path.exists():
        valid_path = processed_root / "upw_valid_predictions.csv"
    if not walk_path.exists():
        walk_path = processed_root / "upw_walkforward_predictions.csv"
    if not drift_path.exists():
        drift_path = processed_root / "upw_fold_drift.csv"

    valid_df = _normalize_predictions(_load_csv(str(valid_path)))
    walk_df = _normalize_predictions(_load_csv(str(walk_path)))
    drift_df = _normalize_drift(_load_csv(str(drift_path)))

    return {
        "run_id": run_id,
        "metrics": metrics,
        "manifest": manifest,
        "params": params,
        "config": config,
        "run_meta": run_meta,
        "valid_df": valid_df,
        "walk_df": walk_df,
        "drift_df": drift_df,
    }


@st.cache_data(show_spinner=False)
def _run_leaderboard(runs_root_str: str) -> pd.DataFrame:
    runs_root = Path(runs_root_str)
    rows: list[dict[str, Any]] = []
    for run_id in _list_runs(runs_root):
        run_root = runs_root / run_id
        metrics = _load_json(str(run_root / "metrics.json"))
        manifest = _load_json(str(run_root / "data_manifest.json"))
        run_meta = _load_json(str(run_root / "run.json"))
        created_at = run_meta.get("created_at")
        rows.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "valid_wape": _metric(metrics, "valid", "wape"),
                "valid_cal_wape": _metric(metrics, "valid_calibrated", "wape"),
                "test_wape": _metric(metrics, "test", "wape"),
                "test_cal_wape": _metric(metrics, "test_calibrated", "wape"),
                "walk_wape": _metric(metrics, "walk_forward", "wape"),
                "naive_valid_wape": _baseline_metric(metrics, "valid", "naive", "wape"),
                "moving_avg_valid_wape": _baseline_metric(metrics, "valid", "moving_avg", "wape"),
                "feature_hash": manifest.get("weekly_features_hash_md5"),
                "week_min": ((manifest.get("week_start_range") or {}).get("min")),
                "week_max": ((manifest.get("week_start_range") or {}).get("max")),
                "unique_weeks": manifest.get("unique_weeks"),
                "unique_drugs": manifest.get("unique_drugs"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("run_id", ascending=False).reset_index(drop=True)


def _choose_prediction_col(df: pd.DataFrame, prefer_calibrated: bool) -> str:
    if prefer_calibrated and "y_pred_cal" in df.columns and df["y_pred_cal"].notna().any():
        return "y_pred_cal"
    return "y_pred"


def _kpi_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, label in SPLITS:
        section = metrics.get(split, {})
        if not isinstance(section, dict):
            continue
        row: dict[str, Any] = {"split": label, "split_key": split}
        for metric_name in METRICS:
            row[metric_name] = section.get(metric_name)
        rows.append(row)
    return pd.DataFrame(rows)


def _weekly_error_trend(df: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    if df.empty or prediction_col not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=["week_start", "y_true", prediction_col]).copy()
    if work.empty:
        return pd.DataFrame()

    work["abs_error"] = (work["y_true"] - work[prediction_col]).abs()
    work["sq_error"] = (work["y_true"] - work[prediction_col]) ** 2
    work["under_flag"] = (work[prediction_col] < work["y_true"]).astype(float)
    denom = (work["y_true"].abs() + work[prediction_col].abs()).replace(0, pd.NA)
    work["smape_row"] = (2 * work["abs_error"]) / denom

    grouped = (
        work.groupby("week_start", as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", "mean"),
            abs_err_sum=("abs_error", "sum"),
            y_sum=("y_true", "sum"),
            smape=("smape_row", "mean"),
            under_forecast_rate=("under_flag", "mean"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    grouped["rmse"] = grouped["rmse"] ** 0.5
    grouped["wape"] = grouped["abs_err_sum"] / grouped["y_sum"].replace(0, pd.NA)
    return grouped


def _scale_01(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    clean = numeric.dropna()
    if clean.empty:
        return pd.Series(0.0, index=numeric.index)
    lo, hi = float(clean.min()), float(clean.max())
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=numeric.index)
    return ((numeric - lo) / (hi - lo)).fillna(0.0)


def _risk_table(df: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    if df.empty or prediction_col not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=["DrugId", "week_start", "y_true", prediction_col]).copy()
    if work.empty:
        return pd.DataFrame()

    work["abs_error"] = (work["y_true"] - work[prediction_col]).abs()
    work["under_flag"] = (work[prediction_col] < work["y_true"]).astype(float)
    work["under_gap"] = (work["y_true"] - work[prediction_col]).clip(lower=0)
    if work["error_rate"].dropna().empty:
        work["error_rate"] = work["abs_error"] / work["y_true"].replace(0, pd.NA)

    latest = (
        work.sort_values(["DrugId", "week_start"])
        .groupby("DrugId", as_index=False)
        .tail(1)[["DrugId", "week_start", "y_true", prediction_col, "spike_score_8"]]
        .rename(
            columns={
                "week_start": "latest_week",
                "y_true": "latest_true",
                prediction_col: "latest_pred",
                "spike_score_8": "latest_spike_score",
            }
        )
    )

    grouped = (
        work.groupby("DrugId", as_index=False)
        .agg(
            genericname=("genericname", "first"),
            saleCategory=("saleCategory", "first"),
            priceCategory=("priceCategory", "first"),
            classified_drug=("classified_drug", "first"),
            n_obs=("y_true", "size"),
            mae=("abs_error", "mean"),
            abs_err_sum=("abs_error", "sum"),
            y_sum=("y_true", "sum"),
            under_forecast_rate=("under_flag", "mean"),
            avg_error_rate=("error_rate", "mean"),
            spike_score_8_max=("spike_score_8", "max"),
        )
        .merge(latest, on="DrugId", how="left")
    )

    grouped["wape"] = grouped["abs_err_sum"] / grouped["y_sum"].replace(0, pd.NA)
    grouped["latest_under_gap"] = (grouped["latest_true"] - grouped["latest_pred"]).clip(lower=0)

    grouped["risk_score"] = (
        0.55 * _scale_01(grouped["wape"])
        + 0.30 * grouped["under_forecast_rate"].fillna(0)
        + 0.15 * _scale_01(grouped["spike_score_8_max"].clip(lower=0))
    )

    cols = [
        "DrugId",
        "genericname",
        "saleCategory",
        "priceCategory",
        "classified_drug",
        "n_obs",
        "mae",
        "wape",
        "avg_error_rate",
        "under_forecast_rate",
        "spike_score_8_max",
        "latest_week",
        "latest_true",
        "latest_pred",
        "latest_under_gap",
        "risk_score",
    ]
    return grouped[cols].sort_values("risk_score", ascending=False).reset_index(drop=True)


def _comparison_metrics(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_key, split_label in [("valid", "Valid"), ("test", "Test"), ("walk_forward", "Walk-forward")]:
        for metric_name in METRICS:
            left = _metric(metrics_a, split_key, metric_name)
            right = _metric(metrics_b, split_key, metric_name)
            delta = (right - left) if (left is not None and right is not None) else None
            if delta is None:
                status = "n/a"
            elif delta < 0:
                status = "improved"
            elif delta > 0:
                status = "regressed"
            else:
                status = "flat"

            rows.append(
                {
                    "split": split_label,
                    "metric": metric_name,
                    "run_a": left,
                    "run_b": right,
                    "delta_b_minus_a": delta,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def _comparison_context(bundle_a: dict[str, Any], bundle_b: dict[str, Any]) -> pd.DataFrame:
    manifest_a = bundle_a["manifest"]
    manifest_b = bundle_b["manifest"]
    params_a = bundle_a["params"]
    params_b = bundle_b["params"]

    rows = [
        {
            "field": "features_hash_md5",
            "run_a": manifest_a.get("weekly_features_hash_md5"),
            "run_b": manifest_b.get("weekly_features_hash_md5"),
        },
        {
            "field": "week_range",
            "run_a": f"{(manifest_a.get('week_start_range') or {}).get('min')} -> {(manifest_a.get('week_start_range') or {}).get('max')}",
            "run_b": f"{(manifest_b.get('week_start_range') or {}).get('min')} -> {(manifest_b.get('week_start_range') or {}).get('max')}",
        },
        {
            "field": "unique_weeks",
            "run_a": manifest_a.get("unique_weeks"),
            "run_b": manifest_b.get("unique_weeks"),
        },
        {
            "field": "unique_drugs",
            "run_a": manifest_a.get("unique_drugs"),
            "run_b": manifest_b.get("unique_drugs"),
        },
        {
            "field": "split_train",
            "run_a": (params_a.get("split_weeks") or {}).get("train"),
            "run_b": (params_b.get("split_weeks") or {}).get("train"),
        },
        {
            "field": "split_valid",
            "run_a": (params_a.get("split_weeks") or {}).get("valid"),
            "run_b": (params_b.get("split_weeks") or {}).get("valid"),
        },
        {
            "field": "split_test",
            "run_a": (params_a.get("split_weeks") or {}).get("test"),
            "run_b": (params_b.get("split_weeks") or {}).get("test"),
        },
    ]
    frame = pd.DataFrame(rows)
    for col in ["run_a", "run_b"]:
        frame[col] = frame[col].fillna("").astype(str)
    return frame


def _guardrail_table(metrics: dict[str, Any], split_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    raw = metrics.get(split_key, {})
    if isinstance(raw, dict):
        rows.append({"model": "xgb_raw", **{metric: raw.get(metric) for metric in METRICS}})

    calibrated = metrics.get(f"{split_key}_calibrated", {})
    if isinstance(calibrated, dict):
        rows.append({"model": "xgb_calibrated", **{metric: calibrated.get(metric) for metric in METRICS}})

    for baseline in ["naive", "moving_avg", "seasonal_naive"]:
        baseline_row = {metric: _baseline_metric(metrics, split_key, baseline, metric) for metric in METRICS}
        rows.append({"model": baseline, **baseline_row})

    frame = pd.DataFrame(rows)
    if "wape" in frame.columns:
        frame = frame.sort_values("wape", ascending=True, na_position="last")
    return frame


def _render_overview(
    bundle: dict[str, Any],
    leaderboard: pd.DataFrame,
    prediction_col_for_walk: str,
) -> None:
    metrics = bundle["metrics"]
    manifest = bundle["manifest"]
    run_meta = bundle["run_meta"]
    walk_df = bundle["walk_df"]

    created_at = str(run_meta.get("created_at") or "Unknown")
    week_range = manifest.get("week_start_range", {}) if isinstance(manifest, dict) else {}
    week_min = week_range.get("min")
    week_max = week_range.get("max")
    unique_weeks = manifest.get("unique_weeks")
    unique_drugs = manifest.get("unique_drugs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run ID", bundle["run_id"])
    c2.metric("Created at", created_at.replace("T", " ").replace("Z", " UTC"))
    c3.metric("Unique weeks", unique_weeks)
    c4.metric("Unique drugs", unique_drugs)
    st.caption(f"Data range: `{week_min}` to `{week_max}`")

    kpi = _kpi_frame(metrics)
    if kpi.empty:
        st.warning("No KPI metrics found for this run.")
    else:
        split_options = kpi["split_key"].tolist()
        split_choice = st.selectbox(
            "KPI split",
            options=split_options,
            index=split_options.index("valid") if "valid" in split_options else 0,
            format_func=lambda s: dict(SPLITS).get(s, s),
        )
        selected = kpi[kpi["split_key"] == split_choice].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{selected['mae']:.3f}" if pd.notna(selected["mae"]) else "n/a")
        c2.metric("RMSE", f"{selected['rmse']:.3f}" if pd.notna(selected["rmse"]) else "n/a")
        c3.metric("WAPE", f"{selected['wape']:.3f}" if pd.notna(selected["wape"]) else "n/a")
        c4.metric("sMAPE", f"{selected['smape']:.3f}" if pd.notna(selected["smape"]) else "n/a")
        st.dataframe(
            kpi.drop(columns=["split_key"]).set_index("split"),
            width="stretch",
            height=260,
        )

    st.subheader("Champion vs challengers (guardrail)")
    split_for_guardrail = st.selectbox("Guardrail split", ["valid", "test"], index=0)
    guardrail = _guardrail_table(metrics, split_for_guardrail)
    if guardrail.empty:
        st.info("No baseline metrics available for this split.")
    else:
        st.dataframe(guardrail, width="stretch", height=220)

    st.subheader("Weekly error trend")
    trend = _weekly_error_trend(walk_df, prediction_col_for_walk)
    if trend.empty:
        st.info("Walk-forward prediction file is empty or missing.")
    else:
        c1, c2 = st.columns(2)
        c1.line_chart(
            trend.set_index("week_start")[["wape", "smape", "under_forecast_rate"]]
        )
        c2.line_chart(
            trend.set_index("week_start")[["mae", "rmse"]]
        )

    st.subheader("Run leaderboard (latest first)")
    if leaderboard.empty:
        st.info("No runs found for comparison.")
    else:
        cols = [
            "run_id",
            "created_at",
            "valid_wape",
            "test_wape",
            "valid_cal_wape",
            "test_cal_wape",
            "naive_valid_wape",
            "moving_avg_valid_wape",
        ]
        display = leaderboard[cols].copy()
        st.dataframe(display, width="stretch", height=280)


def _render_explorer(valid_df: pd.DataFrame, prediction_col: str) -> None:
    if valid_df.empty:
        st.warning("Validation predictions are unavailable.")
        return

    st.caption("Filter by drug metadata and inspect y_true vs model prediction with uncertainty bands.")

    drug_options = sorted(valid_df["DrugId"].dropna().unique().tolist())
    sale_options = sorted(valid_df["saleCategory"].dropna().unique().tolist())
    price_options = sorted(valid_df["priceCategory"].dropna().unique().tolist())
    class_options = sorted(valid_df["classified_drug"].dropna().unique().tolist())

    c1, c2 = st.columns(2)
    selected_drugs = c1.multiselect("DrugId", options=drug_options)
    generic_query = c2.text_input("genericname contains", value="")

    c3, c4, c5 = st.columns(3)
    selected_sale = c3.multiselect("saleCategory", options=sale_options, default=sale_options)
    selected_price = c4.multiselect("priceCategory", options=price_options, default=price_options)
    selected_class = c5.multiselect("classified_drug", options=class_options, default=class_options)

    segment_options = ["low_value", "mid_value", "high_value"]
    selected_segments = st.multiselect(
        "Value segment (derived from y_true tertiles)",
        options=segment_options,
        default=segment_options,
    )

    filtered = valid_df.copy()
    if selected_drugs:
        filtered = filtered[filtered["DrugId"].isin(selected_drugs)]
    if generic_query.strip():
        filtered = filtered[filtered["genericname"].str.contains(generic_query.strip(), case=False, na=False)]
    filtered = filtered[filtered["saleCategory"].isin(selected_sale)]
    filtered = filtered[filtered["priceCategory"].isin(selected_price)]
    filtered = filtered[filtered["classified_drug"].isin(selected_class)]
    filtered = filtered[filtered["value_segment"].isin(selected_segments)]

    r1, r2, r3 = st.columns(3)
    r1.metric("Rows", f"{len(filtered):,}")
    r2.metric("Drugs", f"{filtered['DrugId'].nunique():,}")
    r3.metric("Weeks", f"{filtered['week_start'].nunique():,}")

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    chart_df = filtered.copy()
    chart_df["prediction"] = chart_df[prediction_col]

    grouped = (
        chart_df.groupby("week_start", as_index=False)
        .agg(
            y_true=("y_true", "sum"),
            prediction=("prediction", "sum"),
            y_pred_low=("y_pred_low", "sum"),
            y_pred_high=("y_pred_high", "sum"),
        )
        .sort_values("week_start")
    )
    st.line_chart(
        grouped.set_index("week_start")[["y_true", "prediction", "y_pred_low", "y_pred_high"]]
    )

    st.dataframe(
        filtered.sort_values("week_start", ascending=False)[
            [
                "week_start",
                "DrugId",
                "genericname",
                "saleCategory",
                "priceCategory",
                "classified_drug",
                "value_segment",
                "y_true",
                prediction_col,
                "y_pred_low",
                "y_pred_high",
                "error_rate",
            ]
        ],
        width="stretch",
        height=320,
    )


def _render_risk(valid_df: pd.DataFrame, prediction_col: str) -> None:
    if valid_df.empty:
        st.warning("Validation predictions are unavailable.")
        return

    c1, c2, c3 = st.columns(3)
    top_n = c1.slider("Top N risky drugs", min_value=5, max_value=50, value=15, step=1)
    under_threshold = c2.slider("Under-forecast threshold", min_value=0.10, max_value=0.95, value=0.55, step=0.05)
    spike_threshold = c3.slider("Spike score threshold", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

    risk = _risk_table(valid_df, prediction_col)
    if risk.empty:
        st.warning("Could not compute risk table from selected run.")
        return

    top = risk.head(top_n)
    st.bar_chart(top.set_index("DrugId")[["risk_score"]])
    st.dataframe(top, width="stretch", height=340)

    alerts = risk[
        (risk["under_forecast_rate"] >= under_threshold)
        & (risk["spike_score_8_max"] >= spike_threshold)
        & (risk["latest_under_gap"] > 0)
    ].copy()
    st.subheader("Shortage risk alerts")
    if alerts.empty:
        st.success("No drugs triggered the current alert thresholds.")
    else:
        st.warning(f"{len(alerts)} drugs triggered shortage alerts.")
        st.dataframe(
            alerts[
                [
                    "DrugId",
                    "genericname",
                    "saleCategory",
                    "priceCategory",
                    "wape",
                    "under_forecast_rate",
                    "spike_score_8_max",
                    "latest_true",
                    "latest_pred",
                    "latest_under_gap",
                    "risk_score",
                ]
            ],
            width="stretch",
            height=280,
        )


def _render_drift(drift_df: pd.DataFrame) -> None:
    if drift_df.empty:
        st.warning("No drift diagnostics found.")
        return

    threshold = st.slider(
        "Drift threshold on |mean_pct|",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
    )
    flagged = drift_df[drift_df["mean_pct"].abs() > threshold].copy()

    c1, c2 = st.columns(2)
    c1.metric("Flagged features", f"{len(flagged):,}")
    c2.metric("Total drift features", f"{len(drift_df):,}")

    if flagged.empty:
        st.success("No features exceed the current drift threshold.")
    else:
        st.warning(f"{len(flagged)} features exceed |mean_pct| > {threshold:.2f}.")
        st.dataframe(flagged, width="stretch", height=280)

    st.subheader("Top absolute mean drift")
    top = drift_df.iloc[drift_df["mean_pct"].abs().sort_values(ascending=False).index].head(20)
    st.bar_chart(top.set_index("feature")[["mean_pct"]])
    st.dataframe(drift_df, width="stretch", height=320)


def _render_run_comparison(
    runs_root: Path,
    data_root: Path,
    runs: list[str],
    active_run: str,
) -> None:
    if len(runs) < 2:
        st.info("At least two runs are needed for comparison.")
        return

    default_a = runs[1] if active_run == runs[0] and len(runs) > 1 else runs[0]
    default_b = active_run

    c1, c2 = st.columns(2)
    run_a = c1.selectbox("Run A (baseline)", options=runs, index=runs.index(default_a))
    run_b = c2.selectbox("Run B (candidate)", options=runs, index=runs.index(default_b))

    bundle_a = _load_run_bundle(str(runs_root), str(data_root), run_a)
    bundle_b = _load_run_bundle(str(runs_root), str(data_root), run_b)

    compare = _comparison_metrics(bundle_a["metrics"], bundle_b["metrics"])
    st.dataframe(compare, width="stretch", height=360)

    st.subheader("Split + feature-hash context")
    context = _comparison_context(bundle_a, bundle_b)
    st.dataframe(context, width="stretch", height=250)

    regression_rows = compare[(compare["metric"] == "wape") & (compare["delta_b_minus_a"] > 0)]
    if regression_rows.empty:
        st.success("No WAPE regression detected from Run A to Run B.")
    else:
        bad_splits = ", ".join(regression_rows["split"].tolist())
        st.warning(f"WAPE regressed in: {bad_splits}")

    st.subheader("Guardrail comparison (valid split)")
    guardrail_a = _guardrail_table(bundle_a["metrics"], "valid").set_index("model")
    guardrail_b = _guardrail_table(bundle_b["metrics"], "valid").set_index("model")
    merged = guardrail_a.add_suffix("_a").join(guardrail_b.add_suffix("_b"), how="outer")
    st.dataframe(merged.reset_index(), width="stretch", height=280)


def main() -> None:
    st.set_page_config(page_title="Pharmacy Forecast Dashboard", layout="wide")

    root = _project_root()
    data_root = root / "data"
    runs_root = root / "artifacts" / "runs"
    runs = _list_runs(runs_root)

    st.title("PharmacyStore Forecast Dashboard")
    st.caption(
        "Overview, forecast explorer, risk alerts, drift monitor, and run comparison "
        "for UPW XGBoost outputs."
    )

    if not runs:
        st.error(f"No runs found in {runs_root}.")
        return

    st.sidebar.header("Dashboard Controls")
    active_run = st.sidebar.selectbox("Active run", options=runs, index=0)
    prediction_mode = st.sidebar.radio(
        "Prediction output",
        options=["Raw (Champion)", "Calibrated (Challenger)"],
        index=0,
    )

    bundle = _load_run_bundle(str(runs_root), str(data_root), active_run)
    leaderboard = _run_leaderboard(str(runs_root))
    prefer_calibrated = prediction_mode.startswith("Calibrated")

    valid_df = bundle["valid_df"]
    walk_df = bundle["walk_df"]
    drift_df = bundle["drift_df"]

    prediction_col_valid = _choose_prediction_col(valid_df, prefer_calibrated=prefer_calibrated)
    prediction_col_walk = _choose_prediction_col(walk_df, prefer_calibrated=prefer_calibrated)
    if prefer_calibrated and prediction_col_valid != "y_pred_cal":
        st.sidebar.info("No calibrated predictions in this run file; raw predictions are used.")

    tab_overview, tab_explorer, tab_risk, tab_drift, tab_compare = st.tabs(
        ["Overview", "Forecast Explorer", "Risk & Alert", "Drift Monitor", "Run Comparison"]
    )

    with tab_overview:
        _render_overview(bundle=bundle, leaderboard=leaderboard, prediction_col_for_walk=prediction_col_walk)
    with tab_explorer:
        _render_explorer(valid_df=valid_df, prediction_col=prediction_col_valid)
    with tab_risk:
        _render_risk(valid_df=valid_df, prediction_col=prediction_col_valid)
    with tab_drift:
        _render_drift(drift_df=drift_df)
    with tab_compare:
        _render_run_comparison(runs_root=runs_root, data_root=data_root, runs=runs, active_run=active_run)


if __name__ == "__main__":
    main()
