import os
from binance import Client
import pandas as pd
import logging

from constants import BINANCE_DATA_COLS, DB_DATASETS
from db import create_connection
from log import get_logger
import asyncio


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
    logger = get_logger()
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    klines = await get_historical_klines(symbol, interval)
    interval = "1mo" if interval == "1M" else interval
    klines.to_sql(
        symbol.lower() + "_" + interval, conn, if_exists="replace", index=False
    )
    await logger.log(
        f"Downloaded klines on {symbol} with {interval} interval",
        logging.INFO,
        True,
        True,
    )


def get_all_tickers():
    client = Client()
    data = client.get_all_tickers()
    return data
