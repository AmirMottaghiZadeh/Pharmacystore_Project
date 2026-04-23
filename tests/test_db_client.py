"""Tests for db/client module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pharmacystore.db.client import _build_conn_str, fetch_drugs
from pharmacystore.config import Settings


def test_build_conn_str_complete():
    settings = Settings(
        sql_server="localhost",
        sql_database="testdb",
        sql_username="user",
        sql_password="pass",
        sql_driver="ODBC Driver 18 for SQL Server",
        sql_encrypt=False,
        sql_trust_cert=True,
    )
    
    conn_str = _build_conn_str(settings)
    
    assert "SERVER=localhost" in conn_str
    assert "DATABASE=testdb" in conn_str
    assert "UID=user" in conn_str
    assert "PWD=pass" in conn_str
    assert "Encrypt=no" in conn_str
    assert "TrustServerCertificate=yes" in conn_str


def test_build_conn_str_missing_fields():
    settings = Settings(
        sql_server=None,
        sql_database="testdb",
        sql_username="user",
        sql_password="pass",
    )
    
    with pytest.raises(ValueError, match="Missing DB settings"):
        _build_conn_str(settings)


@patch("pharmacystore.db.client.pd.read_sql")
@patch("pharmacystore.db.client.get_connection")
def test_fetch_drugs(mock_get_conn, mock_read_sql):
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_read_sql.return_value = pd.DataFrame({"ID": [1, 2], "GenericName": ["A", "B"]})
    
    result = fetch_drugs(mock_conn)
    
    assert len(result) == 2
    assert "ID" in result.columns
    assert "GenericName" in result.columns
    mock_read_sql.assert_called_once()
