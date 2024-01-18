from io import BytesIO
import os
from fastapi import UploadFile
import pandas as pd
import numpy as np
import sqlite3

from constants import DB_DATASETS, STREAMING_DEFAULT_CHUNK_SIZE, NullFillStrategy
from config import append_app_data_path
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


def df_fill_nulls(df: pd.DataFrame, column: str, strategy: NullFillStrategy):
    if strategy == NullFillStrategy.ZERO:
        df[column].fillna(0, inplace=True)

    elif strategy == NullFillStrategy.MEAN:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

    elif strategy == NullFillStrategy.CLOSEST:
        if df[column].isnull().any():
            ffill = df[column].ffill()
            bfill = df[column].bfill()

            ffill_mask = df[column].ffill().notnull()
            bfill_mask = df[column].bfill().notnull()

            index_series = pd.Series(df.index, index=df.index)

            ffill_dist = (
                index_series[ffill_mask].reindex_like(df, method="ffill") - df.index
            )
            bfill_dist = df.index - index_series[bfill_mask].reindex_like(
                df, method="bfill"
            )

            df[column] = np.where(ffill_dist <= bfill_dist, ffill, bfill)


class PythonCode:
    INDENT = "    "
    DATASET_SYMBOL = "dataset"
    EDIT_COLUMN_DEFAULT = f"def run_python({DATASET_SYMBOL}):\n{INDENT}"
    SAVE_STATEMENT = "with sqlite3.connect(AppConstants.DB_DATASETS) as conn:"

    @classmethod
    def append_code(cls, dataset_name: str, code: str):
        return (
            cls.EDIT_COLUMN_DEFAULT
            + code
            + f"\n{cls.INDENT}"
            + cls.SAVE_STATEMENT
            + f"\n{cls.INDENT}{cls.INDENT}"
            + f'{cls.DATASET_SYMBOL}.to_sql("{dataset_name}", conn, if_exists="replace", index=False)'
            + f'\nrun_python(read_dataset_to_mem("{dataset_name}"))'
        )
