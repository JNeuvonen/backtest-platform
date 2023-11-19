import os
from binance import Client
import pandas as pd

from constants import BINANCE_DATA_COLS, DB_DATASETS
from db import create_connection
from server import get_logger


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
logger = get_logger()


def get_historical_klines(symbol, interval):
    client = Client()
    start_time = "1 Jan, 2017"
    klines = []

    while True:
        new_klines = client.get_historical_klines(
            symbol, interval, start_time, limit=1000
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


def save_historical_klines(symbol, interval):
    logger = get_logger()
    logger.info(f"Initiating downloading klines on {symbol} with {interval} interval")
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    klines = get_historical_klines(symbol, interval)
    klines.to_sql(symbol + interval, conn, if_exists="replace", index=False)
    logger.info(f"Succesfully fetched klines on {symbol} with {interval} interval")
