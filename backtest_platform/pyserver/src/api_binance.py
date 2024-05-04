import os
import logging
import asyncio
import pandas as pd

from binance import Client
from constants import BINANCE_DATA_COLS, AppConstants, BinanceDataCols, DomEventChannels
from db import create_connection
from log import LogExceptionContext, get_logger
from query_dataset import DatasetQuery
from utils import get_binance_dataset_tablename


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")


async def get_historical_klines(symbol, interval):
    client = Client()
    start_time = "1 Jan, 2017"
    klines = []

    while True:
        new_klines = await asyncio.to_thread(
            client.get_historical_klines, symbol, interval, start_time, limit=1000
        )
        if not new_klines:
            break

        klines += new_klines
        start_time = int(new_klines[-1][0]) + 1

    df = pd.DataFrame(klines, columns=BINANCE_DATA_COLS)
    df.drop(["ignore", "kline_close_time"], axis=1, inplace=True)
    df["kline_open_time"] = pd.to_numeric(df["kline_open_time"])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("kline_open_time", inplace=True)
    return df


def non_async_get_historical_klines(symbol, interval):
    client = Client()
    start_time = "1 Jan, 2017"
    klines = []

    while True:
        new_klines = client.get_historical_klines(
            symbol=symbol, interval=interval, start_str=start_time, limit=1000
        )
        if not new_klines:
            break

        klines += new_klines
        start_time = int(new_klines[-1][0]) + 1

    df = pd.DataFrame(klines, columns=BINANCE_DATA_COLS)
    df.drop(["ignore", "kline_close_time"], axis=1, inplace=True)
    df["kline_open_time"] = pd.to_numeric(df["kline_open_time"])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("kline_open_time", inplace=True)
    return df


async def save_historical_klines(symbol, interval, send_msg_to_fe=True):
    with LogExceptionContext(notification_duration=60000):
        logger = get_logger()
        datasets_conn = create_connection(AppConstants.DB_DATASETS)
        klines = await get_historical_klines(symbol, interval)
        table_name = get_binance_dataset_tablename(symbol, interval)

        table_exists_query = (
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        cursor = datasets_conn.cursor()
        cursor.execute(table_exists_query)
        if cursor.fetchone():
            logger.log(f"Table {table_name} already exists.", logging.INFO)
            DatasetQuery.delete_entry_by_dataset_name(table_name)

        klines.to_sql(table_name, datasets_conn, if_exists="replace", index=False)
        DatasetQuery.create_dataset_entry(
            dataset_name=table_name,
            timeseries_column="kline_open_time",
            price_column=BinanceDataCols.CLOSE_PRICE,
            symbol=symbol,
            interval=interval,
        )

        if send_msg_to_fe is True:
            logger.log(
                f"Downloaded klines on {symbol} with {interval} interval",
                logging.INFO,
                True,
                True,
                DomEventChannels.REFETCH_ALL_DATASETS.value,
            )


def non_async_save_historical_klines(symbol, interval, send_msg_to_fe=True):
    with LogExceptionContext(notification_duration=60000):
        logger = get_logger()
        datasets_conn = create_connection(AppConstants.DB_DATASETS)
        klines = non_async_get_historical_klines(symbol, interval)
        table_name = get_binance_dataset_tablename(symbol, interval)

        table_exists_query = (
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        cursor = datasets_conn.cursor()
        cursor.execute(table_exists_query)
        if cursor.fetchone():
            logger.log(f"Table {table_name} already exists.", logging.INFO)
            DatasetQuery.delete_entry_by_dataset_name(table_name)

        klines.to_sql(table_name, datasets_conn, if_exists="replace", index=False)
        DatasetQuery.create_dataset_entry(table_name, "kline_open_time")

        if send_msg_to_fe is True:
            logger.log(
                f"Downloaded klines on {symbol} with {interval} interval",
                logging.INFO,
                True,
                True,
                DomEventChannels.REFETCH_ALL_DATASETS.value,
            )


def get_all_tickers():
    with LogExceptionContext(notification_duration=60000, logging_level=logging.ERROR):
        client = Client()
        data = client.get_all_tickers()
        return data
