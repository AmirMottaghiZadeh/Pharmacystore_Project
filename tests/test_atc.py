import pandas as pd

from pharmacystore.features.atc import classify_atc_codes


def test_classify_atc_codes_matches_who_csv(tmp_path) -> None:
    who_csv = tmp_path / "who.csv"
    who_csv.write_text(
        "atc_code,atc_name\nA01A,Alpha\nB02B,Beta\n", encoding="utf-8"
    )
    drugs_df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "AtcCode": ["A01A123", " b02b ", None],
        }
    )

    out = classify_atc_codes(drugs_df, who_csv_path=str(who_csv))
    assert out.loc[0, "classified_code"] == "A01A"
    assert out.loc[0, "atc_name"] == "Alpha"
    assert out.loc[1, "classified_code_clean"] == "B02B"
    assert out.loc[1, "atc_name"] == "Beta"
    assert pd.isna(out.loc[2, "atc_name"])
