from typing import Optional

import pandas as pd
import pyodbc

from pharmacystore.config import Settings, get_settings


def _build_conn_str(settings: Settings) -> str:
    missing = [
        name
        for name in ["sql_server", "sql_database", "sql_username", "sql_password"]
        if not getattr(settings, name)
    ]
    if missing:
        raise ValueError(
            "Missing DB settings: "
            + ", ".join(missing)
            + ". Set PHARMACYSTORE_SQL_* values in .env or env vars."
        )

    encrypt = "yes" if settings.sql_encrypt else "no"
    trust_cert = "yes" if settings.sql_trust_cert else "no"
    return (
        f"DRIVER={{{settings.sql_driver}}};"
        f"SERVER={settings.sql_server};"
        f"DATABASE={settings.sql_database};"
        f"UID={settings.sql_username};"
        f"PWD={settings.sql_password};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_cert};"
    )


def get_connection(settings: Optional[Settings] = None) -> pyodbc.Connection:
    """Open a new database connection."""
    active_settings = settings or get_settings()
    return pyodbc.connect(_build_conn_str(active_settings))


def _run_query(query: str, conn: Optional[pyodbc.Connection] = None) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame, closing the connection if we opened it."""
    should_close = conn is None
    active_conn = conn or get_connection()
    df = pd.read_sql(query, active_conn)
    if should_close:
        active_conn.close()
    return df


def fetch_drugs(conn: Optional[pyodbc.Connection] = None) -> pd.DataFrame:
    query = """
    SELECT ID, GenericName, BrandName, AtcCode , IsDrugs , Hospital , IsSumDrug , IsOTCDRUG , IsTarkibi , SubGroupID
    FROM dr_Drugs
    """
    return _run_query(query, conn)


def fetch_prescription_detail(conn: Optional[pyodbc.Connection] = None) -> pd.DataFrame:
    query = """
    SELECT PrID, DrugId, PacketQuantity, LogDateTime, SalePrice
    FROM dr_PrescriptionDetailLog
    """
    return _run_query(query, conn)


def fetch_factor_detail(conn: Optional[pyodbc.Connection] = None) -> pd.DataFrame:
    query = """
    SELECT DrugID, PacketQuantity, Qnt, SalePrice AS FactorSalePrice, KharidPrice
    FROM dr_FactorDetail
    """
    return _run_query(query, conn)
