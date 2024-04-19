import math
import threading
from io import BytesIO
import os
from fastapi import UploadFile
import pandas as pd
import sqlite3

from sqlalchemy.inspection import inspect

from constants import (
    DB_DATASETS,
    STREAMING_DEFAULT_CHUNK_SIZE,
    CandleSize,
)
from config import append_app_data_path
from log import LogExceptionContext
from query_trainjob import TrainJobQuery


def convert_val_split_str_to_arr(val_split_str: str):
    parts = val_split_str.split(",")
    return [int(parts[0]), int(parts[1])]


def rm_file(path):
    if os.path.isfile(path):
        os.remove(path)


def add_to_datasets_db(df: pd.DataFrame, table_name: str):
    with sqlite3.connect(append_app_data_path(DB_DATASETS)) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def remove_all_csv_files(directory):
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            os.remove(os.path.join(directory, file))


async def read_file_to_dataframe(
    file: UploadFile, chunk_size: int = STREAMING_DEFAULT_CHUNK_SIZE
) -> pd.DataFrame:
    with LogExceptionContext():
        chunks = []

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)

        file_bytes = b"".join(chunks)
        return pd.read_csv(BytesIO(file_bytes))


class PythonCode:
    INDENT = "    "
    DATASET_SYMBOL = "dataset"
    COLUMN_SYMBOL = "column_name"
    FUNC_EXEC_ON_COLUMN = f"def run_on_column({DATASET_SYMBOL}, {COLUMN_SYMBOL}):"
    FUNC_EXEC_ON_DATASET = f"def run_on_dataset({DATASET_SYMBOL}):"
    OPEN_CONNECTION = "with sqlite3.connect(AppConstants.DB_DATASETS) as conn:"
    DATASET_CODE_EXAMPLE = "dataset = get_dataset()"
    COLUMN_CODE_EXAMPLE = f"{COLUMN_SYMBOL} = get_column()"

    @classmethod
    def on_column(cls, dataset_name: str, column_name: str, code: str):
        code = "\n".join(cls.INDENT + line for line in code.split("\n"))
        code = (
            code.replace(cls.DATASET_CODE_EXAMPLE, "")
            .replace(cls.COLUMN_CODE_EXAMPLE, "")
            .rstrip()
        )
        return (
            cls.FUNC_EXEC_ON_COLUMN
            + f"\n{code}"
            + f"\n{cls.INDENT}{cls.OPEN_CONNECTION}"
            + f'\n{cls.INDENT}{cls.INDENT}{cls.DATASET_SYMBOL}.to_sql("{dataset_name}", conn, if_exists="replace", index=False)'
            + f'\nrun_on_column(read_dataset_to_mem("{dataset_name}"), "{column_name}")'
        )

    @classmethod
    def on_dataset(cls, dataset_name: str, code: str):
        code = "\n".join(cls.INDENT + line for line in code.split("\n"))
        code = (
            code.replace(cls.DATASET_CODE_EXAMPLE, "")
            .replace(cls.COLUMN_CODE_EXAMPLE, "")
            .rstrip()
        )
        return (
            cls.FUNC_EXEC_ON_DATASET
            + f"\n{code}"
            + f"\n{cls.INDENT}{cls.OPEN_CONNECTION}"
            + f'\n{cls.INDENT}{cls.INDENT}{cls.DATASET_SYMBOL}.to_sql("{dataset_name}", conn, if_exists="replace", index=False)'
            + f'\nrun_on_dataset(read_dataset_to_mem("{dataset_name}"))'
        )


class GlobalSymbols:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalSymbols, cls).__new__(cls)
            cls.GLOBALS_ON_APP_LAUNCH = set(globals().keys())
        return cls._instance

    @staticmethod
    def cleanup_globals():
        current_globals = set(globals().keys())
        new_globals = current_globals - GlobalSymbols.GLOBALS_ON_APP_LAUNCH

        for global_var in new_globals:
            del globals()[global_var]


global_symbols = GlobalSymbols()


def run_in_thread(fn, *args, **kwargs):
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
    thread.start()
    return thread


def on_shutdown_cleanup():
    TrainJobQuery.on_shutdown_cleanup()


def to_dict(obj):
    return {attr: getattr(obj, attr) for attr in vars(obj)}


def get_col(datasetColStrFormat: str):
    parts = datasetColStrFormat.split("_")
    return parts[1]


def get_is_invalid_number(num):
    if num == math.inf:
        return True
    elif num == -math.inf:
        return True
    elif math.isnan(num):
        return True

    return False


def base_model_to_dict(obj):
    return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}


def contains_infinity(dct):
    return any(
        math.isinf(value) for value in dct.values() if isinstance(value, (float, int))
    )


def replace_infinities(dct):
    for k, v in dct.items():
        if isinstance(v, (float, int)) and math.isinf(v):
            dct[k] = 0
    return dct


def is_string(var):
    return isinstance(var, str)


def get_binance_dataset_tablename(symbol: str, interval: str):
    interval = "1mo" if interval == "1M" else interval
    table_name = symbol.lower() + "_" + interval
    return table_name


def read_file_to_string(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def get_periods_per_year(candle_interval: str):
    if candle_interval == CandleSize.ONE_MINUTE:
        return 365 * 24 * 60
    if candle_interval == CandleSize.THREE_MINUTES:
        return 365 * 24 * 20

    if candle_interval == CandleSize.FIVE_MINUTES:
        return 365 * 24 * 12

    if candle_interval == CandleSize.FIFTEEN_MINUTES:
        return 365 * 24 * 4

    if candle_interval == CandleSize.THIRTY_MINUTES:
        return 365 * 24 * 2

    if candle_interval == CandleSize.ONE_HOUR:
        return 365 * 24

    if candle_interval == CandleSize.TWO_HOURS:
        return 365 * 12

    if candle_interval == CandleSize.FOUR_HOURS:
        return 365 * 6

    if candle_interval == CandleSize.SIX_HOURS:
        return 365 * 4

    if candle_interval == CandleSize.EIGHT_HOURS:
        return 365 * 3

    if candle_interval == CandleSize.TWELVE_HOURS:
        return 365 * 2

    if candle_interval == CandleSize.ONE_DAY:
        return 365

    if candle_interval == CandleSize.THREE_DAYS:
        return 122

    if candle_interval == CandleSize.ONE_WEEK:
        return 52

    if candle_interval == CandleSize.ONE_MONTH:
        return 12

    raise ValueError("Invalid candle size")
