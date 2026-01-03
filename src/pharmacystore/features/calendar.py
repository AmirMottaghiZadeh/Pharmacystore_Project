"""
Build Gregorian equivalents for Persian holidays and aggregate official holidays by ISO week.

Outputs:
- data/external/holidays/persian_holidays_gregorian.csv : original rows with a computed gregorian_date column.
- data/external/holidays/weekly_official_holidays.csv  : number of official holiday days per ISO week.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd


JALALI_MONTH_DAYS = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29]


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> date:
    """Convert a Jalali (Shamsi) date to a Gregorian date."""
    jy -= 979
    jm -= 1
    jd -= 1

    j_day_no = 365 * jy + jy // 33 * 8 + ((jy % 33) + 3) // 4
    j_day_no += sum(JALALI_MONTH_DAYS[:jm]) + jd

    g_day_no = j_day_no + 79

    gy = 1600 + 400 * (g_day_no // 146097)
    g_day_no %= 146097

    leap = True
    if g_day_no >= 36525:
        g_day_no -= 1
        gy += 100 * (g_day_no // 36524)
        g_day_no %= 36524

        if g_day_no >= 365:
            g_day_no += 1
        else:
            leap = False

    gy += 4 * (g_day_no // 1461)
    g_day_no %= 1461

    if g_day_no >= 366:
        leap = False
        g_day_no -= 1
        gy += g_day_no // 365
        g_day_no %= 365

    gd = g_day_no + 1
    g_month_days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while gd > g_month_days[gm]:
        gd -= g_month_days[gm]
        gm += 1

    return date(gy, gm + 1, gd)


def add_gregorian_date(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a gregorian_date column computed from Jalali year/month/day."""
    df["gregorian_date"] = df.apply(
        lambda row: jalali_to_gregorian(int(row["year"]), int(row["month"]), int(row["day"])),
        axis=1,
    )
    return df


def compute_weekly_official_counts(dates: Iterable[date]) -> pd.DataFrame:
    """Count unique official holiday days per project week (Monday–Sunday).

    Missing weeks in the range are included with a zero count.
    """
    df = pd.DataFrame({"gregorian_date": pd.to_datetime(list(dates))}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["week_period", "week_start", "week_end", "official_holiday_days"])

    # Align with the rest of the project (weeks anchored to Sunday, start on Monday).
    df["week_period"] = df["gregorian_date"].dt.to_period("W-SUN")
    df["week_start"] = df["week_period"].dt.start_time.dt.normalize()
    df["week_end"] = df["week_start"] + pd.Timedelta(days=6)

    weekly = (
        df.groupby("week_period", as_index=False)
        .agg(
            week_start=("week_start", "first"),
            week_end=("week_end", "first"),
            official_holiday_days=("gregorian_date", lambda x: x.dt.date.nunique()),
        )
    )

    full_periods = pd.period_range(start=weekly["week_period"].min(), end=weekly["week_period"].max(), freq="W-SUN")
    full_weeks = pd.DataFrame({"week_period": full_periods})
    full_weeks["week_start"] = full_weeks["week_period"].dt.start_time
    full_weeks["week_end"] = full_weeks["week_period"].dt.end_time
    full_weeks["week_start"] = full_weeks["week_start"].dt.normalize()
    full_weeks["week_end"] = full_weeks["week_end"].dt.normalize()

    merged = full_weeks.merge(weekly[["week_period", "official_holiday_days"]], on="week_period", how="left")
    merged["official_holiday_days"] = merged["official_holiday_days"].fillna(0).astype(int)

    merged = merged.sort_values("week_start").reset_index(drop=True)
    return merged[["week_period", "week_start", "week_end", "official_holiday_days"]]


def build_datasets() -> None:
    project_root = Path(__file__).resolve().parents[3]
    holiday_dir = project_root / "data" / "external" / "holidays"
    src_csv = holiday_dir / "persian_holidays.csv"
    gregorian_csv = holiday_dir / "persian_holidays_gregorian.csv"
    weekly_csv = holiday_dir / "weekly_official_holidays.csv"

    if not src_csv.exists():
        raise FileNotFoundError(f"Expected source CSV at {src_csv}")

    holidays = pd.read_csv(src_csv)
    holidays = add_gregorian_date(holidays)
    holidays.to_csv(gregorian_csv, index=False)

    official_dates = holidays.loc[holidays["is_holiday"] == 1, "gregorian_date"]
    weekly_counts = compute_weekly_official_counts(official_dates)
    weekly_counts.to_csv(weekly_csv, index=False)

    print(f"Saved holiday rows with Gregorian dates to: {gregorian_csv}")
    print(f"Saved weekly official holiday counts ({len(weekly_counts)} rows) to: {weekly_csv}")


if __name__ == "__main__":
    build_datasets()
