from io import BytesIO
import os
from fastapi import UploadFile
import pandas as pd
import sqlite3

from constants import DB_DATASETS, STREAMING_DEFAULT_CHUNK_SIZE
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
