from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def merge_drug_prescription_factor(
    drugs_df: pd.DataFrame,
    presc_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    sort_key: str = "DrugID",
    use_factor_agg: bool = False,
) -> pd.DataFrame:
    """Merge drug, prescription, and factor data with optional sorting key."""
    drugs = drugs_df.rename(columns={"ID": "DrugID"})
    presc = presc_df.rename(columns={"DrugId": "DrugID"})

    if sort_key.lower() == "prid":
        presc_clean = presc.sort_values(by=["PrID", "DrugID", "LogDateTime"])
        presc_clean = presc_clean.drop_duplicates(subset=["PrID", "DrugID"], keep="first")
        final_sort = ["PrID", "LogDateTime"]
    else:
        presc_clean = presc.sort_values(by=["DrugID", "PrID", "LogDateTime"])
        presc_clean = presc_clean.drop_duplicates(subset=["DrugID", "PrID"], keep="first")
        final_sort = ["DrugID", "LogDateTime"]

    merged = presc_clean.merge(drugs, on="DrugID", how="left")
    if use_factor_agg and not factor_df.empty:
        factor_agg = (
            factor_df.groupby("DrugID", as_index=False)
            .agg(
                {
                    "PacketQuantity": "sum",
                    "Qnt": "sum",
                    "FactorSalePrice": "mean",
                    "KharidPrice": "mean",
                }
            )
        )
        merged = merged.merge(factor_agg, on="DrugID", how="left")
    merged = merged.sort_values(final_sort).reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["DrugID", "PrID"], keep="first")
    return merged


def compute_sales_price_stats(prescription_df: pd.DataFrame, k: int = 10) -> dict:
    """Fit sales/price scaling statistics on a training subset to avoid leakage."""
    df = prescription_df.copy()
    df = df[df["SalePrice"] > 0]
    agg = (
        df.groupby("DrugId")
        .agg(
            UniquePackets=("PrID", "nunique"),
            AvgSalePrice=("SalePrice", "mean"),
        )
        .reset_index()
    )
    min_val = agg["UniquePackets"].min()
    max_val = agg["UniquePackets"].max()
    non_zero_prices = agg.loc[agg["AvgSalePrice"] > 0, "AvgSalePrice"]
    min_price = 0 if non_zero_prices.empty else non_zero_prices.min()
    max_price = agg["AvgSalePrice"].max()

    return {
        "k": k,
        "min_unique": float(min_val),
        "max_unique": float(max_val),
        "min_price": float(min_price),
        "max_price": float(max_price),
    }


def calculate_sales_and_price_categories(
    prescription_df: pd.DataFrame,
    stats: Optional[dict] = None,
    k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    """Scale sales metrics and classify drugs by sales volume and price."""
    df = prescription_df.copy()
    df = df[df["SalePrice"] > 0]

    result = (
        df.groupby("DrugId")
        .agg(
            UniquePackets=("PrID", "nunique"),
            TotalPackets=("PacketQuantity", "sum"),
            AvgSalePrice=("SalePrice", "mean"),
            SalePrice=("SalePrice", "max"),
        )
        .reset_index()
    )

    used_stats = stats or compute_sales_price_stats(df, k=k)
    k_val = used_stats["k"]
    min_val = used_stats["min_unique"]
    max_val = used_stats["max_unique"]
    min_price = used_stats["min_price"]
    max_price = used_stats["max_price"]

    # Scale UniquePackets to [0, 1]
    if max_val == min_val:
        result["Scale"] = 0.0
    else:
        result["Scale"] = (result["UniquePackets"] - min_val) / (max_val - min_val)

    x1 = 1 / k_val
    x2 = 1 / (2 * k_val)
    x3 = 1 / (8 * k_val)
    x4 = 1 / (8 * (k_val**2))

    conditions_sales = [
        result["Scale"] > x1,
        (result["Scale"] <= x1) & (result["Scale"] > x2),
        (result["Scale"] <= x2) & (result["Scale"] > x3),
        (result["Scale"] <= x3) & (result["Scale"] > x4),
        result["Scale"] <= x4,
    ]

    choices_sales = [
        "VeryBest-selling",
        "Best-selling",
        "Mid-selling",
        "Least-selling",
        "VeryLeast-selling",
    ]

    result["SalesCategory"] = np.select(conditions_sales, choices_sales, default="Unknown")

    # Scale AvgSalePrice to [0, 1], ignoring zero min
    if max_price == min_price:
        result["PriceScale"] = 0.0
    else:
        result["PriceScale"] = (result["AvgSalePrice"] - min_price) / (max_price - min_price)

    result["PriceScale"] = result["PriceScale"].clip(lower=0)

    conditions_price = [
        result["PriceScale"] > x1,
        (result["PriceScale"] <= x1) & (result["PriceScale"] > x2),
        (result["PriceScale"] <= x2) & (result["PriceScale"] > x3),
        (result["PriceScale"] <= x3) & (result["PriceScale"] > x4),
        result["PriceScale"] <= x4,
    ]

    choices_price = [
        "VeryExpensive",
        "Expensive",
        "Mid_Price",
        "InExpensive",
        "VeryInexpensive",
    ]

    result["PriceCategory"] = np.select(conditions_price, choices_price, default="Unknown")

    target_labels = {"VeryBest-selling", "Best-selling"}
    selected_drugs_df = result[result["SalesCategory"].isin(target_labels)][
        ["DrugId", "SalesCategory"]
    ]
    selected_drugs = selected_drugs_df.to_dict(orient="records")

    return result, selected_drugs_df, selected_drugs, used_stats


def build_weekly_unique_packets(
    prescription_df: pd.DataFrame,
    target_drug_ids: Iterable,
    holiday_dates_shamsi: Optional[Iterable] = None,
    holiday_weekly_path: Optional[str] = "data/external/holidays/weekly_official_holidays.csv",
) -> pd.DataFrame:
    """Calculate weekly unique packet counts per DrugId with rich calendrical features.

    Parameters
    ----------
    prescription_df : pd.DataFrame
        Raw prescription detail data.
    target_drug_ids : Iterable
        Drugs to keep; use None to keep all.
    """
    df = prescription_df.copy()
    df["LogDateTime"] = pd.to_datetime(df["LogDateTime"], errors="coerce")
    df = df[df["LogDateTime"].notna()]
    df = df[df["SalePrice"] > 0]

    if target_drug_ids is not None:
        df = df[df["DrugId"].isin(set(target_drug_ids))]

    df["week_start"] = df["LogDateTime"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    weekly_unique = (
        df.groupby(["DrugId", "week_start"])
        .agg(UniquePackets=("PrID", "nunique"))
        .reset_index()
        .sort_values(["DrugId", "week_start"])
    )

    def fill_missing_weeks(group: pd.DataFrame) -> pd.DataFrame:
        full_weeks = pd.date_range(
            start=group["week_start"].min(),
            end=group["week_start"].max(),
            freq="W-MON",
        )
        out = (
            group.set_index("week_start")
            .reindex(full_weeks, fill_value=0)
            .rename_axis("week_start")
            .reset_index()
        )
        out["DrugId"] = group.name
        out["UniquePackets"] = out["UniquePackets"].astype(int)
        return out[["DrugId", "week_start", "UniquePackets"]]

    weekly_complete = (
        weekly_unique.groupby("DrugId", group_keys=False)
        .apply(fill_missing_weeks, include_groups=False)
        .sort_values(["DrugId", "week_start"])
        .reset_index(drop=True)
    )
    weekly_complete["UPW"] = weekly_complete["UniquePackets"]

    if holiday_weekly_path:
        holiday_path = Path(holiday_weekly_path)
        if not holiday_path.is_absolute():
            holiday_path = PROJECT_ROOT / holiday_path
        if holiday_path.exists():
            weekly_holidays = pd.read_csv(holiday_path)
            if "week_start" in weekly_holidays.columns:
                weekly_holidays["week_start"] = pd.to_datetime(
                    weekly_holidays["week_start"], errors="coerce"
                )
                weekly_holidays = (
                    weekly_holidays[["week_start", "official_holiday_days"]]
                    .dropna(subset=["week_start"])
                    .drop_duplicates(subset=["week_start"])
                )
                weekly_complete = weekly_complete.merge(weekly_holidays, on="week_start", how="left")
        if "official_holiday_days" not in weekly_complete.columns:
            weekly_complete["official_holiday_days"] = 0
        weekly_complete["official_holiday_days"] = (
            weekly_complete["official_holiday_days"].fillna(0).astype(int)
        )

    # Week- and month-level time features on week_start anchor
    week_numbers = weekly_complete["week_start"].dt.isocalendar().week.astype(int)
    weekly_complete["week_of_year"] = week_numbers
    weekly_complete["month"] = weekly_complete["week_start"].dt.month
    weekly_complete["quarter"] = weekly_complete["week_start"].dt.quarter
    month_key = weekly_complete["week_start"].dt.to_period("M")
    month_first = weekly_complete.groupby(month_key)["week_start"].transform("min")
    month_last = weekly_complete.groupby(month_key)["week_start"].transform("max")
    weekly_complete["is_month_start"] = weekly_complete["week_start"] == month_first
    weekly_complete["is_month_end"] = weekly_complete["week_start"] == month_last
    weekly_complete["is_quarter_start"] = weekly_complete["week_start"].dt.is_quarter_start
    weekly_complete["is_quarter_end"] = weekly_complete["week_start"].dt.is_quarter_end
    weekly_complete["week_sinus"] = np.sin(2 * np.pi * week_numbers / 52)

    weekly_complete = (
        weekly_complete.sort_values(["DrugId", "week_of_year", "week_start"])
        .reset_index(drop=True)
    )

    return weekly_complete


def add_weekly_rolling_features(weekly_df: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    """Add rolling statistics and change metrics to a weekly UPW DataFrame."""
    df = weekly_df.copy()
    if "UPW" not in df and "UniquePackets" in df.columns:
        df["UPW"] = df["UniquePackets"]

    def _apply_rolls(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("week_start").copy()
        g["DrugId"] = group.name
        g["UPW_lag1"] = g["UPW"].shift(1)
        g["UPW_lag2"] = g["UPW"].shift(2)
        g["UPW_lag3"] = g["UPW"].shift(3)
        g["UPW_lag4"] = g["UPW"].shift(4)
        g["UPW_lag6"] = g["UPW"].shift(6)
        lag_series = g["UPW_lag1"]

        g["UPW_rollmean_2"] = lag_series.rolling(window=2, min_periods=2).mean()
        g["UPW_rollmean_4"] = lag_series.rolling(window=4, min_periods=4).mean()
        g["UPW_rollmean_6"] = lag_series.rolling(window=6, min_periods=6).mean()
        g["UPW_rollstd_3"] = lag_series.rolling(window=3, min_periods=3).std()
        g["UPW_rollstd_6"] = lag_series.rolling(window=6, min_periods=6).std()
        g["UPW_rollmin"] = lag_series.rolling(window=3, min_periods=3).min()
        g["UPW_rollmax"] = lag_series.rolling(window=3, min_periods=3).max()

        g["diff_rate"] = lag_series.pct_change()
        g["avg_point"] = (lag_series + lag_series.shift(1)) / 2
        roll_mean_12 = lag_series.rolling(window=12, min_periods=12).mean()
        roll_std_12 = lag_series.rolling(window=12, min_periods=12).std()
        g["Z_score"] = (lag_series - roll_mean_12) / (roll_std_12 + epsilon)

        # Guard against inf from zero divisors and drop rows without full history
        g = g.replace([np.inf, -np.inf], np.nan)
        history_cols = [
            "UPW_lag1",
            "UPW_lag2",
            "UPW_lag3",
            "UPW_lag4",
            "UPW_lag6",
            "UPW_rollmean_2",
            "UPW_rollmean_4",
            "UPW_rollmean_6",
            "UPW_rollstd_3",
            "UPW_rollstd_6",
            "UPW_rollmin",
            "UPW_rollmax",
            "diff_rate",
            "avg_point",
            "Z_score",
        ]
        g = g.dropna(subset=history_cols)
        return g

    enriched = df.groupby("DrugId", group_keys=False).apply(_apply_rolls, include_groups=False)
    return enriched.reset_index(drop=True)
