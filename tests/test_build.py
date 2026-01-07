import numpy as np
import pandas as pd

from pharmacystore.features import build as build_mod


def test_merge_drug_prescription_factor_dedupes_and_sorts() -> None:
    drugs_df = pd.DataFrame({"ID": [1, 2], "Name": ["A", "B"]})
    presc_df = pd.DataFrame(
        {
            "DrugId": [1, 1, 2],
            "PrID": [100, 100, 200],
            "LogDateTime": [
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-03"),
                pd.Timestamp("2023-01-01"),
            ],
        }
    )
    factor_df = pd.DataFrame()

    merged = build_mod.merge_drug_prescription_factor(
        drugs_df, presc_df, factor_df, sort_key="DrugID"
    )
    assert merged.shape[0] == 2
    assert merged["DrugID"].tolist() == [1, 2]


def test_merge_drug_prescription_factor_adds_factor_agg() -> None:
    drugs_df = pd.DataFrame({"ID": [1], "Name": ["A"]})
    presc_df = pd.DataFrame(
        {
            "DrugId": [1],
            "PrID": [100],
            "LogDateTime": [pd.Timestamp("2023-01-02")],
        }
    )
    factor_df = pd.DataFrame(
        {
            "DrugID": [1, 1],
            "PacketQuantity": [2, 3],
            "Qnt": [1, 2],
            "FactorSalePrice": [10.0, 20.0],
            "KharidPrice": [5.0, 7.0],
        }
    )

    merged = build_mod.merge_drug_prescription_factor(
        drugs_df, presc_df, factor_df, sort_key="DrugID", use_factor_agg=True
    )
    row = merged.iloc[0]
    assert row["PacketQuantity"] == 5
    assert row["Qnt"] == 3
    assert row["FactorSalePrice"] == 15.0
    assert row["KharidPrice"] == 6.0


def test_compute_sales_price_stats_ignores_nonpositive() -> None:
    df = pd.DataFrame(
        {
            "DrugId": [1, 1, 2],
            "PrID": [10, 11, 12],
            "SalePrice": [100, 0, 50],
        }
    )

    stats = build_mod.compute_sales_price_stats(df, k=5)
    assert stats["k"] == 5
    assert stats["min_unique"] == 1.0
    assert stats["max_unique"] == 1.0
    assert stats["min_price"] == 50.0
    assert stats["max_price"] == 100.0


def test_calculate_sales_and_price_categories_uses_stats() -> None:
    rows = [
        {"DrugId": 1, "PrID": pr_id, "PacketQuantity": 1, "SalePrice": 100}
        for pr_id in range(1, 13)
    ]
    rows.append({"DrugId": 2, "PrID": 100, "PacketQuantity": 1, "SalePrice": 10})
    df = pd.DataFrame(rows)

    stats = build_mod.compute_sales_price_stats(df, k=10)
    result, selected_df, selected_list, used_stats = build_mod.calculate_sales_and_price_categories(
        df, stats=stats
    )
    assert used_stats is stats
    assert "SalesCategory" in result.columns
    assert 1 in selected_df["DrugId"].tolist()
    assert any(item["DrugId"] == 1 for item in selected_list)


def test_build_weekly_unique_packets_fills_missing_weeks(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "DrugId": [1, 1, 1],
            "PrID": [10, 11, 12],
            "LogDateTime": ["2023-01-02", "2023-01-04", "2023-01-16"],
            "SalePrice": [100, 100, 100],
        }
    )
    missing_path = tmp_path / "missing.csv"
    out = build_mod.build_weekly_unique_packets(
        df, target_drug_ids=[1], holiday_weekly_path=str(missing_path)
    )
    assert out["week_start"].tolist() == [
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2023-01-09"),
        pd.Timestamp("2023-01-16"),
    ]
    assert out["UniquePackets"].tolist() == [2, 0, 1]
    assert out["official_holiday_days"].tolist() == [0, 0, 0]


def test_add_weekly_rolling_features_produces_history() -> None:
    week_start = pd.date_range("2023-01-02", periods=20, freq="W-MON")
    df = pd.DataFrame(
        {
            "DrugId": [1] * len(week_start),
            "week_start": week_start,
            "UniquePackets": np.arange(1, len(week_start) + 1),
        }
    )

    out = build_mod.add_weekly_rolling_features(df)
    assert not out.empty
    for col in ["UPW_lag1", "UPW_rollmean_2", "UPW_rollstd_3", "Z_score"]:
        assert col in out.columns
        assert out[col].notna().all()
