from pharmacystore.db.client import fetch_drugs, get_connection
from pharmacystore.features.atc import classify_atc_codes


def main() -> None:
    with get_connection() as conn:
        generic_df = fetch_drugs(conn)

    generic_df = classify_atc_codes(generic_df)
    defined_mask = generic_df["AtcCode"].fillna("").str.strip() != ""

    print(f"All rows: {len(generic_df)}, with AtcCode: {defined_mask.sum()}")
    print(
        generic_df.loc[
            defined_mask, ["ID", "GenericName", "AtcCode", "classified_code", "atc_name"]
        ].head()
    )


if __name__ == "__main__":
    main()
