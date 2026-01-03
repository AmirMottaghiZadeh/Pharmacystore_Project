from pharmacystore.db.client import fetch_prescription_detail, get_connection
from pharmacystore.features.build import calculate_sales_and_price_categories


def main() -> None:
    with get_connection() as conn:
        df = fetch_prescription_detail(conn)

    result, selected_drugs_df, selected_drugs, _ = calculate_sales_and_price_categories(df)

    print("\nSelected drugs (DrugId with SalesCategory):")
    print(selected_drugs_df.to_string(index=False))
    print("\nSelected drugs as list of dicts:")
    print(selected_drugs)

    print(result.head(20))

    print("\nCounts by SalesCategory:")
    print(result["SalesCategory"].value_counts())

    print("\nCounts by PriceCategory:")
    print(result["PriceCategory"].value_counts())

    output_file = "data/processed/selected_drugs.csv"
    selected_drugs_df.to_csv(output_file, index=False)
    print(f"\nSaved selected drugs to {output_file}")


if __name__ == "__main__":
    main()
