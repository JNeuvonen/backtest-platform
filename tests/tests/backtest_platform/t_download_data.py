import pandas as pd
import sys
import os
from binance import Client

from tests.backtest_platform.t_conf import SERVER_SOURCE_DIR

sys.path.append(SERVER_SOURCE_DIR)
from constants import BINANCE_DATA_COLS


def download_historical_binance_data(symbol, interval, file_path):
    if os.path.exists(file_path):
        return None

    start_time = "1 Jan, 2017"
    klines = []
    client = Client()

    while True:
        new_klines = client.get_historical_klines(
            symbol, interval, start_time, limit=1000
        )
        if not new_klines:
            break

        klines += new_klines
        start_time = int(new_klines[-1][0]) + 1

    df = pd.DataFrame(klines, columns=BINANCE_DATA_COLS)
    df["kline_open_time"] = pd.to_numeric(df["kline_open_time"])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("kline_open_time", inplace=True)

    df.to_csv(file_path, index=False)
    return None
