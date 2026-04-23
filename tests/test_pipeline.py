"""Tests for pipeline module."""

import pandas as pd
import pytest

from pharmacystore.pipeline import collapse_to_generic


def test_collapse_to_generic_aggregates_brands():
    weekly_df = pd.DataFrame({
        "genericname": ["Aspirin", "Aspirin", "Ibuprofen"],
        "brandname": ["BrandA", "BrandB", "BrandC"],
        "week_start": ["2023-01-02", "2023-01-02", "2023-01-02"],
        "UPW": [10, 20, 30],
        "UniquePackets": [5, 10, 15],
        "price": [100, 200, 300],
    })
    
    result = collapse_to_generic(weekly_df)
    
    assert len(result) == 2  # Two unique generics
    aspirin_row = result[result["genericname"] == "Aspirin"].iloc[0]
    assert aspirin_row["UPW"] == 30  # Sum of 10 + 20
    assert aspirin_row["UniquePackets"] == 15  # Sum of 5 + 10
    assert "brandname" not in result.columns


def test_collapse_to_generic_handles_empty_genericname():
    weekly_df = pd.DataFrame({
        "genericname": ["", "  ", "Valid"],
        "week_start": ["2023-01-02", "2023-01-02", "2023-01-02"],
        "UPW": [10, 20, 30],
        "UniquePackets": [5, 10, 15],
    })
    
    result = collapse_to_generic(weekly_df)
    
    assert "XXXX" in result["genericname"].values
    assert "Valid" in result["genericname"].values
