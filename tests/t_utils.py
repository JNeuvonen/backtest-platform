from contextlib import contextmanager
import os
import sys
from typing import List
import pandas as pd
import requests
from tests.t_conf import SERVER_SOURCE_DIR
from tests.t_constants import URL, DatasetMetadata
from tests.t_context import t_file


sys.path.append(SERVER_SOURCE_DIR)

from config import append_app_data_path
from constants import BINANCE_DATA_COLS
from utils import add_to_datasets_db
from orm import DatasetQuery


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
    DatasetQuery.create_dataset_entry(dataset.name, dataset.timeseries_col)


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


def create_model_body(
    name: str,
    target_col: str,
    drop_cols: List[str],
    null_fill_strategy,
    model: str,
    hyper_params_and_optimizer_code: str,
    validation_split: List[int],
):
    return {
        "name": name,
        "target_col": target_col,
        "drop_cols": drop_cols,
        "null_fill_strategy": null_fill_strategy,
        "model": model,
        "hyper_params_and_optimizer_code": hyper_params_and_optimizer_code,
        "validation_split": validation_split,
    }


def create_train_job_body(
    num_epochs: int,
    save_model_after_every_epoch: bool,
    backtest_on_val_set: bool,
    enter_trade_criteria: str,
    exit_trade_criteria: str,
):
    return {
        "num_epochs": num_epochs,
        "save_model_after_every_epoch": save_model_after_every_epoch,
        "backtest_on_val_set": backtest_on_val_set,
        "enter_trade_criteria": enter_trade_criteria,
        "exit_trade_criteria": exit_trade_criteria,
    }


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

    @staticmethod
    def get_dataset_models(dataset_name: str):
        with Req("get", URL.get_dataset_models(dataset_name)) as res:
            res_json = res.json()
            return res_json["data"]

    @staticmethod
    def get_model_by_name(model_name: str):
        with Req("get", URL.get_model_by_name(model_name)) as res:
            res_json = res.json()
            return res_json["model"]


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
    def exec_python_on_col(dataset_name, column_name, body):
        with Req(
            "post", URL.exec_python_on_col(dataset_name, column_name), json=body
        ) as res:
            return res.json()

    @staticmethod
    def exec_python_on_dataset(dataset_name, body):
        with Req("post", URL.exec_python_on_dataset(dataset_name), json=body) as res:
            return res.json()

    @staticmethod
    def create_model(dataset_name, body):
        with Req("post", URL.create_model(dataset_name), json=body) as res:
            return res.json()

    @staticmethod
    def create_train_job(model_name: str, body):
        with Req("post", URL.create_train_job(model_name), json=body) as res:
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
