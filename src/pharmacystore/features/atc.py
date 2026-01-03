from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def classify_atc_codes(
    drugs_df: pd.DataFrame, who_csv_path: str = "data/external/who_atc_ddd.csv"
) -> pd.DataFrame:
    """Build classified ATC codes and map to ATC names from WHO reference CSV."""
    generic_df = drugs_df.copy()

    atc_clean = generic_df["AtcCode"].fillna("").astype(str).str.strip()
    generic_df["classified_code"] = atc_clean.str[:4]
    generic_df["classified_code_clean"] = (
        generic_df["classified_code"].astype(str).str.strip().str.upper()
    )

    who_path = Path(who_csv_path)
    if not who_path.is_absolute():
        who_path = PROJECT_ROOT / who_path
    who_df = pd.read_csv(who_path)
    who_df["atc_code_clean"] = who_df["atc_code"].astype(str).str.strip().str.upper()
    who_df = who_df.drop_duplicates(subset="atc_code_clean", keep="first")

    merged = generic_df.merge(
        who_df[["atc_code_clean", "atc_name"]],
        how="left",
        left_on="classified_code_clean",
        right_on="atc_code_clean",
    )
    return merged
