from io import BytesIO
import os
from fastapi import UploadFile
import pandas as pd
import numpy as np
import sqlite3

from constants import (
    DB_DATASETS,
    STREAMING_DEFAULT_CHUNK_SIZE,
    AppConstants,
    NullFillStrategy,
)
from config import append_app_data_path
from dataset import read_dataset_to_mem
from log import LogExceptionContext


def rm_file(path):
    if os.path.isfile(path):
        os.remove(path)


def add_to_datasets_db(df: pd.DataFrame, table_name: str):
    with sqlite3.connect(append_app_data_path(DB_DATASETS)) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


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
