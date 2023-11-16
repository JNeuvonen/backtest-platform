from sqlite3 import Connection
import pandas as pd
from db import create_connection
from constants import BINANCE_DATA_COLS, DATASETS_DB
import shutil
import os


def combine(
    conn: (Connection | None),
    path_to_dir: str,
    pair: str,
    spot: str,
    dataseries_interval: str,
    candle_interval: str,
):
    if conn is None:
        return

    dfs = []

    for root, _, files in os.walk(path_to_dir):
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, names=BINANCE_DATA_COLS, skiprows=1)
            df.drop(columns=["kline_close_time", "ignore"], inplace=True)
            col_prefix = pair + "_" + spot + "_"
            df = df.add_prefix(col_prefix)
            df = df.rename(columns={col_prefix + "kline_open_time": "kline_open_time"})
            df.columns = df.columns.str.lower()
            dfs.append(df)

    combined_dir = os.path.join(path_to_dir, "merged")

    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)
    else:
        shutil.rmtree(combined_dir)
        os.makedirs(combined_dir)

    result = pd.concat(dfs, axis=0)
    result = result.sort_values(by=["kline_open_time"])
    table_name = pair + "_" + dataseries_interval + "_" + spot + "_" + candle_interval
    table_name = table_name.lower()
    result.to_sql(table_name, conn, if_exists="replace", index=False)


if __name__ == "__main__":
    conn = create_connection(DATASETS_DB)
    combine(
        conn,
        "../../../data/spot/klines/IOTAUSDT/1h/",
        "IOTAUSDT",
        "spot",
        "monthly",
        "1h",
    )
