from pharmacystore.db.client import fetch_prescription_detail, get_connection
from pharmacystore.features.build import (
    build_weekly_unique_packets,
    calculate_sales_and_price_categories,
)


def main() -> None:
    with get_connection() as conn:
        df = fetch_prescription_detail(conn)

    _, selected_drugs_df, _, _ = calculate_sales_and_price_categories(df)
    weekly_complete = build_weekly_unique_packets(df, selected_drugs_df["DrugId"])

    print(f"Total weekly rows (with filled gaps): {len(weekly_complete)}")
    print(weekly_complete.head())


if __name__ == "__main__":
    main()
