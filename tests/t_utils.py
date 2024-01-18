from contextlib import contextmanager
import os
import pandas as pd
import requests
from pyserver.src.config import append_app_data_path
from pyserver.src.constants import BINANCE_DATA_COLS
from pyserver.src.db import DatasetUtils
from pyserver.src.utils import add_to_datasets_db
from tests.t_constants import URL, DatasetMetadata
from tests.t_context import t_file


@contextmanager
def Req(method, url, **kwargs):
    with requests.request(method, url, **kwargs) as response:
        response.raise_for_status()
        yield response


def t_fetch_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def t_get_timeseries_col(table_item):
    return table_item["timeseries_col"]


def add_object_to_add_cols_payload(payload_arr, table_name, cols):
    payload_arr.append({"table_name": table_name, "columns": cols})


def read_csv_to_df(path):
    return pd.read_csv(append_app_data_path(path))


def t_add_binance_dataset_to_db(dataset: DatasetMetadata):
    df = read_csv_to_df(dataset.path)
    df.columns = BINANCE_DATA_COLS
    add_to_datasets_db(df, dataset.name)
    DatasetUtils.create_db_utils_entry(dataset.name, dataset.timeseries_col)


def t_generate_big_dataframe(data: DatasetMetadata, target_size_bytes: int):
    current_size = 0
    dataframe_list = []

    while current_size <= target_size_bytes:
        with t_file(data.path) as file:
            df = pd.read_csv(file)
            dataframe_list.append(df)
            current_size += os.path.getsize(file.name)
            if current_size >= target_size_bytes:
                break

    big_dataframe = pd.concat(dataframe_list, ignore_index=True)
    return big_dataframe


class Fetch:
    @staticmethod
    def get_tables():
        with Req("get", URL.t_get_tables()) as res:
            return res.json()["tables"]

    @staticmethod
    def get_all_tables_and_columns():
        with Req("get", URL.t_get_all_columns()) as res:
            return res.json()["table_col_map"]

    @staticmethod
    def get_dataset_by_name(dataset_name: str):
        with Req("get", URL.t_get_dataset_by_name(dataset_name)) as res:
            return res.json()["dataset"]

    @staticmethod
    def get_dataset_col_info(dataset_name: str, column_name: str):
        with Req("get", URL.get_column_detailed_info(dataset_name, column_name)) as res:
            res_json = res.json()
            return res_json["column"], res_json["timeseries_col"]


class Post:
    @staticmethod
    def rename_column(dataset_name: str, body):
        with Req("post", URL.t_get_rename_column(dataset_name), json=body) as res:
            return res.json()

    @staticmethod
    def add_columns(dataset_name: str, body, null_fill_strategy: str = "NONE"):
        with Req(
            "post",
            URL.add_columns_to_dataset(dataset_name, null_fill_strategy),
            json=body,
        ) as res:
            return res.json()

    @staticmethod
    def exec_python(body):
        with Req("post", URL.exec_python(), json=body) as res:
            return res.json()


class Put:
    @staticmethod
    def update_dataset_name(dataset_name: str, body):
        with Req("put", URL.t_update_dataset_name(dataset_name), json=body) as res:
            return res.json()

    @staticmethod
    def update_timeseries_col(dataset_name: str, body):
        with Req("put", URL.update_timeseries_col(dataset_name), json=body) as res:
            return res.json()


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
