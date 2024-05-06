import sqlite3
import pandas as pd

from typing import Optional
from binance_utils import fetch_binance_klines
from constants import strategy_name_to_db_file
from schema.data_transformation import DataTransformationQuery
from utils import (
    calculate_timestamp_for_kline_fetch,
    file_exists,
    get_current_timestamp_ms,
)


def read_dataset_to_mem(db_path: str, table_name: str):
    with sqlite3.connect(db_path) as conn:
        query = f"SELECT * FROM {table_name}"
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


def initiate_dataset(data_fetcher):
    db_path = strategy_name_to_db_file(data_fetcher.strategy_name)

    curr_time_ms = get_current_timestamp_ms()

    klines = fetch_binance_klines(
        data_fetcher.symbol,
        data_fetcher.interval,
        calculate_timestamp_for_kline_fetch(
            data_fetcher.num_required_klines, data_fetcher.kline_size_ms
        ),
    )

    data_transformations = DataTransformationQuery.get_transformations_by_strategy(
        data_fetcher.strategy_id
    )


def append_new_data_to_db(data_fetcher):
    db_path = strategy_name_to_db_file(data_fetcher.strategy_name)

    if file_exists(db_path) is False:
        initiate_dataset(data_fetcher)
        return

    latest_kline_open_time = read_latest_kline_open_time(
        db_path, data_fetcher.strategy_name
    )

    if latest_kline_open_time is None:
        return

    latest_klines = fetch_binance_klines(
        data_fetcher.symbol, data_fetcher.interval, int(latest_kline_open_time) + 1
    )
