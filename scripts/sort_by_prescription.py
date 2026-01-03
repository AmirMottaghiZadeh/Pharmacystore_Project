from pharmacystore.db.client import (
    fetch_drugs,
    fetch_factor_detail,
    fetch_prescription_detail,
    get_connection,
)
from pharmacystore.features.build import merge_drug_prescription_factor


def main() -> None:
    with get_connection() as conn:
        df_drugs = fetch_drugs(conn)
        df_presc = fetch_prescription_detail(conn)
        df_factor = fetch_factor_detail(conn)

    df_all = merge_drug_prescription_factor(df_drugs, df_presc, df_factor, sort_key="PrID")
    print(df_all.head(50))


if __name__ == "__main__":
    main()
