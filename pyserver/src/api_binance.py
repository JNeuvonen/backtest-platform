import os
import logging
import asyncio
import pandas as pd

from binance import Client
from constants import BINANCE_DATA_COLS, DB_DATASETS, DomEventChannels
from db import DatasetUtils, create_connection
from log import LogExceptionContext, get_logger


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


async def save_historical_klines(symbol, interval):
    with LogExceptionContext(notification_duration=60000):
        logger = get_logger()
        datasets_conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
        klines = await get_historical_klines(symbol, interval)
        interval = "1mo" if interval == "1M" else interval

        table_name = symbol.lower() + "_" + interval
        klines.to_sql(table_name, datasets_conn, if_exists="replace", index=False)
        await DatasetUtils.create_db_utils_entry(table_name, "kline_open_time")
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
