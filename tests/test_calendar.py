from datetime import date

import pandas as pd

from pharmacystore.features import calendar as calendar_mod


def test_jalali_to_gregorian_nowruz() -> None:
    assert calendar_mod.jalali_to_gregorian(1402, 1, 1) == date(2023, 3, 21)


def test_add_gregorian_date() -> None:
    df = pd.DataFrame({"year": [1402], "month": [1], "day": [1]})
    out = calendar_mod.add_gregorian_date(df)
    assert out.loc[0, "gregorian_date"] == date(2023, 3, 21)


def test_compute_weekly_official_counts_fills_missing() -> None:
    dates = [date(2023, 1, 2), date(2023, 1, 4), date(2023, 1, 16)]
    result = calendar_mod.compute_weekly_official_counts(dates)
    assert result["official_holiday_days"].tolist() == [2, 0, 1]
    assert result["week_start"].tolist() == [
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2023-01-09"),
        pd.Timestamp("2023-01-16"),
    ]
