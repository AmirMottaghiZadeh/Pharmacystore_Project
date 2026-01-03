from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pharmacystore.db.client import (
    fetch_drugs,
    fetch_factor_detail,
    fetch_prescription_detail,
    get_connection,
)
from pharmacystore.config import Settings, get_settings
from pharmacystore.features.atc import classify_atc_codes
from pharmacystore.features.build import (
    add_weekly_rolling_features,
    build_weekly_unique_packets,
    calculate_sales_and_price_categories,
    compute_sales_price_stats,
    merge_drug_prescription_factor,
)
from pharmacystore.run_utils import set_global_seed


def collapse_to_generic(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all brands under each generic per week; sums UPW/UniquePackets, keeps first meta."""
    df = weekly_df.copy()
    # Normalize genericname: strip whitespace, replace empty with placeholder
    if "genericname" in df.columns:
        gen = df["genericname"].astype(str).str.strip()
        gen = gen.replace("", pd.NA)
        df["genericname"] = gen.fillna("XXXX")
    else:
        df["genericname"] = "XXXX"
    if "brandname" in df.columns:
        df = df.drop(columns=["brandname"])

    key_cols = ["genericname", "week_start"]
    sum_cols = {"UPW", "UniquePackets"}
    agg_map = {}
    for col in df.columns:
        if col in key_cols:
            continue
        if col in sum_cols:
            agg_map[col] = "sum"
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg_map[col] = "first"
        else:
            agg_map[col] = "first"

    grouped = df.groupby(key_cols, as_index=False).agg(agg_map)
    grouped["DrugId"] = grouped["genericname"]
    return grouped


def run_pipeline(settings: Settings | None = None) -> pd.DataFrame:
    """Pull data from SQL, build merged DataFrames, and display quick previews."""
    active_settings = settings or get_settings()
    set_global_seed(active_settings.random_seed)

    with get_connection(active_settings) as conn:
        drugs_df = fetch_drugs(conn)
        presc_df = fetch_prescription_detail(conn)
        factor_df = fetch_factor_detail(conn)

    merged_by_drug = merge_drug_prescription_factor(
        drugs_df, presc_df, factor_df, sort_key="DrugID", use_factor_agg=False
    )
    merged_by_prescription = merge_drug_prescription_factor(
        drugs_df, presc_df, factor_df, sort_key="PrID", use_factor_agg=False
    )

    presc_df["LogDateTime"] = pd.to_datetime(presc_df["LogDateTime"], errors="coerce")
    presc_df = presc_df[presc_df["LogDateTime"].notna()].sort_values("LogDateTime")
    split_idx = int(len(presc_df) * 0.8)
    presc_fit = presc_df.iloc[:split_idx].copy()

    sales_stats = compute_sales_price_stats(presc_fit)
    sales_price_df, selected_drugs_df, selected_drugs, _ = calculate_sales_and_price_categories(
        presc_df, stats=sales_stats
    )
    atc_classified_df = classify_atc_codes(drugs_df)

    weekly_features_df = build_weekly_unique_packets(presc_df, selected_drugs_df["DrugId"])

    sale_price_meta = sales_price_df[["DrugId", "SalesCategory", "PriceCategory"]].rename(
        columns={"SalesCategory": "saleCategory", "PriceCategory": "priceCategory"}
    )
    drug_meta = (
        drugs_df.rename(
            columns={
                "ID": "DrugId",
                "GenericName": "genericname",
                "BrandName": "brandname",
                "IsDrugs": "isdrug",
                "IsOTCDRUG": "isotcdrug",
                "Hospital": "hospital",
                "IsTarkibi": "isTarkibi",
                "SubGroupID": "SubgroupID",
            }
        )[
            [
                "DrugId",
                "genericname",
                "brandname",
                "isdrug",
                "isotcdrug",
                "hospital",
                "isTarkibi",
                "SubgroupID",
            ]
        ]
    )
    classified_meta = atc_classified_df[["ID", "classified_code"]].rename(
        columns={"ID": "DrugId", "classified_code": "classified_drug"}
    )

    weekly_features_df = (
        weekly_features_df.merge(sale_price_meta, on="DrugId", how="left")
        .merge(drug_meta, on="DrugId", how="left")
        .merge(classified_meta, on="DrugId", how="left")
        .sort_values(["week_start", "DrugId"])
        .reset_index(drop=True)
    )
    weekly_features_df = collapse_to_generic(weekly_features_df)
    weekly_features_df = add_weekly_rolling_features(weekly_features_df)

    # Reorder to place generic/brand names right after the date column
    cols = list(weekly_features_df.columns)
    for col in ["genericname", "brandname"]:
        if col in cols:
            cols.remove(col)
    insert_at = cols.index("week_start") + 1 if "week_start" in cols else 1
    reinserts = [c for c in ["genericname", "brandname"] if c in weekly_features_df.columns]
    cols[insert_at:insert_at] = reinserts
    weekly_features_df = weekly_features_df[cols]

    # Round float columns to two decimal places
    float_cols = weekly_features_df.select_dtypes(include="float").columns
    weekly_features_df[float_cols] = weekly_features_df[float_cols].round(2)

    final_weekly_df = weekly_features_df.copy()

    weekly_features_path = Path(active_settings.data_dir) / "processed" / "weekly_features.csv"
    weekly_features_path.parent.mkdir(parents=True, exist_ok=True)
    final_weekly_df.to_csv(weekly_features_path, index=False)

    print("\n=== Merged (sorted by DrugID) sample ===")
    print(merged_by_drug.head())

    print("\n=== Merged (sorted by PrID) sample ===")
    print(merged_by_prescription.head())

    print("\n=== Sales scaling sample ===")
    print(sales_price_df.head())
    print("\nSelected drugs (top-selling labels):")
    print(selected_drugs_df.head())

    print("\n=== ATC classification sample ===")
    print(atc_classified_df[["ID", "GenericName", "AtcCode", "classified_code", "atc_name"]].head())

    print("\n=== Weekly features sample ===")
    print(final_weekly_df.head(10))
    print(f"\nTotal weekly rows: {len(final_weekly_df)}")
    print(f"Selected drugs (dict): {selected_drugs[:5]}")
    print(f"\nSaved weekly features to {weekly_features_path}")
    return final_weekly_df


def main() -> None:
    parser = argparse.ArgumentParser(description="PharmacyStore pipeline runner")
    parser.add_argument(
        "--run-tag",
        help="Optional tag appended to the run folder name (safe chars: a-zA-Z0-9-_).",
        default=None,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Build weekly features from SQL data.")
    subparsers.add_parser("train", help="Train the main XGBoost model.")
    subparsers.add_parser("train-baseline", help="Train the baseline XGBoost model.")
    subparsers.add_parser("full", help="Run pipeline + train the main model.")

    args = parser.parse_args()
    settings = get_settings()

    if args.command == "run":
        run_pipeline(settings=settings)
        return

    if args.command == "train":
        from pharmacystore.models.train_xgb import main as train_main

        train_main(settings=settings, run_tag=args.run_tag)
        return

    if args.command == "train-baseline":
        from pharmacystore.models.train_xgb_baseline import main as train_baseline_main

        train_baseline_main(settings=settings, run_tag=args.run_tag)
        return

    if args.command == "full":
        from pharmacystore.models.train_xgb import main as train_main

        df = run_pipeline(settings=settings)
        train_main(settings=settings, data=df, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
