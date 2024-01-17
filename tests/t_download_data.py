import pandas as pd
import sys
import os
from binance import Client

from tests.t_conf import SERVER_SOURCE_DIR
from tests.t_constants import FixturePaths

sys.path.append(SERVER_SOURCE_DIR)
from constants import BINANCE_DATA_COLS
from config import append_app_data_path


def download_historical_binance_data(symbol, interval, name_of_file):
    output_csv = append_app_data_path(
        FixturePaths.BINANCE_DOWNLOADED.format(name_of_file)
    )
    if os.path.exists(output_csv):
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
    df.drop(["ignore", "kline_close_time"], axis=1, inplace=True)
    df["kline_open_time"] = pd.to_numeric(df["kline_open_time"])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("kline_open_time", inplace=True)

    df.to_csv(output_csv, index=False)
    return None
