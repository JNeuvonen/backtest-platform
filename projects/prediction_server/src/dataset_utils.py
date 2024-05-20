import sqlite3
import pandas as pd

from typing import Optional


def read_dataset_to_mem(engine, table_name: str):
    with engine.connect() as conn:
        query = f'SELECT * FROM "{table_name}"'
        df = pd.read_sql_query(query, conn)
        return df


def read_latest_row(engine, table_name: str):
    with engine.connect() as conn:
        query = f"""
        SELECT * FROM "{table_name}"
        WHERE "kline_open_time" = (
            SELECT MAX("kline_open_time") FROM "{table_name}"
        )
        """
        df = pd.read_sql_query(query, conn)
        return df


def check_table_exists(db_path: str, table_name: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        result = conn.execute(query).fetchone()
        return result is not None


def read_latest_kline_open_time(db_path: str, table_name: str) -> Optional[int]:
    with sqlite3.connect(db_path) as conn:
        query = f"SELECT kline_open_time FROM {table_name} ORDER BY kline_open_time DESC LIMIT 1"
        result = conn.execute(query).fetchone()
        return result[0] if result else None
