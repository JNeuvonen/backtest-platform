from contextlib import contextmanager
import os
import sys
from typing import List
import pandas as pd
from pandas.core.algorithms import mode
import requests
from tests.backtest_platform.t_conf import SERVER_SOURCE_DIR
from tests.backtest_platform.t_constants import URL, DatasetMetadata
from tests.backtest_platform.t_context import t_file


sys.path.append(SERVER_SOURCE_DIR)

from config import append_app_data_path
from constants import BINANCE_DATA_COLS
from utils import add_to_datasets_db
from query_dataset import DatasetQuery


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


def read_csv_to_df(path, header=False):
    return pd.read_csv(append_app_data_path(path), header=0 if header else None)


def t_add_binance_dataset_to_db(dataset: DatasetMetadata):
    df = read_csv_to_df(dataset.path)
    df.columns = BINANCE_DATA_COLS
    add_to_datasets_db(df, dataset.name)
    DatasetQuery.create_dataset_entry(
        dataset.name,
        dataset.timeseries_col,
        dataset.target_col,
        dataset.price_col,
        dataset.symbol,
        dataset.interval,
    )


def t_add_custom_dataset_to_db(dataset: DatasetMetadata):
    df = read_csv_to_df(dataset.path, header=True)
    add_to_datasets_db(df, dataset.name)
    DatasetQuery.create_dataset_entry(
        dataset.name,
        dataset.timeseries_col,
        dataset.target_col,
        dataset.price_col,
        dataset.symbol,
        dataset.interval,
    )


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
    drop_cols: List[str],
    null_fill_strategy,
    model: str,
    hyper_params_and_optimizer_code: str,
    validation_split: List[int],
    scale_target: bool = False,
    scaling_strategy=2,
    drop_cols_on_train: List[str] = ["kline_open_time"],
):
    return {
        "name": name,
        "drop_cols": drop_cols,
        "null_fill_strategy": null_fill_strategy,
        "model": model,
        "hyper_params_and_optimizer_code": hyper_params_and_optimizer_code,
        "validation_split": validation_split,
        "scale_target": scale_target,
        "scaling_strategy": scaling_strategy,
        "drop_cols_on_train": drop_cols_on_train,
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


def gen_data_transformations(dataset_name: str):
    transformation_results = []

    # Execute and store the result of the Aroon calculation
    aroon_result = Post.exec_python_on_dataset(
        dataset_name,
        body={
            "code": """
import numpy as np
import pandas as pd

def calculate_aroon(df, column='close_price', periods=25):
    aroon_up = 100 * df[column].rolling(window=periods, min_periods=0).apply(
        lambda x: float(np.argmax(x) + 1) / periods * 100, raw=True)
    aroon_down = 100 * df[column].rolling(window=periods, min_periods=0).apply(
        lambda x: float(np.argmin(x) + 1) / periods * 100, raw=True)

    df[f'Aroon_Up_{periods}_{column}'] = aroon_up
    df[f'Aroon_Down_{periods}_{column}'] = aroon_down

periods = 25
column = "close_price"
calculate_aroon(dataset, column=column, periods=periods)
    """
        },
    )
    transformation_results.append(aroon_result)

    # Execute and store the result of the Moving Average calculation
    ma_result = Post.exec_python_on_dataset(
        dataset_name,
        body={
            "code": """
def calculate_ma(df, column='close_price', periods=[50]):
    for period in periods:
        ma_label = f"MA_{period}_{column}"
        df[ma_label] = df[column].rolling(window=period).mean()

periods = [50]
column = "close_price"
calculate_ma(dataset, column=column, periods=periods)
    """
        },
    )
    transformation_results.append(ma_result)

    # Execute and store the result of the RSI calculation
    rsi_result = Post.exec_python_on_dataset(
        dataset_name,
        body={
            "code": """
def calculate_rsi(df, column='open_price', periods=[14]):
    for period in periods:
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df_label = f"RSI_{period}_{column}"
        df[df_label] = 100 - (100 / (1 + rs))

periods = [30]
column = "MA_50_close_price"
calculate_rsi(dataset, column=column, periods=periods)
    """
        },
    )
    transformation_results.append(rsi_result)

    return transformation_results


def gen_ml_based_test_data_transformations(dataset_name: str):
    transformation_results = []

    aroon_result = """
import numpy as np
import pandas as pd

def calculate_aroon(df, column='close_price', periods=25):
    aroon_up = df[column].rolling(window=periods, min_periods=0).apply(
        lambda x: float(np.argmax(x) + 1) / periods * 100, raw=True)
    aroon_down = df[column].rolling(window=periods, min_periods=0).apply(
        lambda x: float(np.argmin(x) + 1) / periods * 100, raw=True)

    df[f'AROON_UP_CUST_{periods}_{column}'] = aroon_up
    df[f'AROON_DOWN_CUST_{periods}_{column}'] = aroon_down

# Usage example:
periods = 25
column = "close_price"
calculate_aroon(dataset, column=column, periods=periods)
    """
    transformation_results.append(aroon_result)

    ma_result = Post.exec_python_on_dataset(
        dataset_name,
        body={
            "code": """
def calculate_ma(df, column='close_price', periods=[50]):
    for period in periods:
        ma_label = f"MA_{period}_{column}"
        df[ma_label] = df[column].rolling(window=period).mean()

periods = [50]
column = "close_price"
calculate_ma(dataset, column=column, periods=periods)
    """
        },
    )
    transformation_results.append(ma_result)

    # Execute and store the result of the RSI calculation
    rsi_result = Post.exec_python_on_dataset(
        dataset_name,
        body={
            "code": """
def calculate_rsi(df, column='open_price', periods=[14]):
    for period in periods:
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df_label = f"RSI_{period}_{column}"
        df[df_label] = 100 - (100 / (1 + rs))

periods = [30]
column = "MA_50_close_price"
calculate_rsi(dataset, column=column, periods=periods)
    """
        },
    )
    transformation_results.append(rsi_result)

    target_result = Post.exec_python_on_dataset(
        dataset_name,
        body={
            "code": """
def create_targets(df, column='close_price', shifts=[7, 14, 30]):
    for shift in shifts:
        target_label = f"target_{shift}"
        df[target_label] = df[column].shift(-shift) / df[column]

shifts = [7]
create_targets(dataset, column='close_price', shifts=shifts)
    """
        },
    )
    transformation_results.append(target_result)

    return transformation_results


def create_backtest_body(
    price_column: str,
    epoch_nr: int,
    enter_trade_cond: str,
    exit_trade_cond: str,
    dataset_name: str,
):
    return {
        "price_col": price_column,
        "epoch_nr": epoch_nr,
        "enter_trade_cond": enter_trade_cond,
        "exit_trade_cond": exit_trade_cond,
        "dataset_name": dataset_name,
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
    def get_model_by_id(model_id: int):
        with Req("get", URL.get_model_by_id(model_id)) as res:
            res_json = res.json()
            return res_json["model"]

    @staticmethod
    def all_metadata_by_model_id(model_id: int):
        with Req("get", URL.get_all_metadata_by_model_id(model_id)) as res:
            res_json = res.json()
            return res_json["data"]

    @staticmethod
    def get_dataset_pagination(dataset_name: str, page: int, page_size: int):
        with Req(
            "get", URL.get_dataset_pagination(dataset_name, page, page_size)
        ) as res:
            return res.json()["data"]

    @staticmethod
    def get_datasets_manual_backtests(dataset_id: int):
        with Req("get", URL.get_datasets_manual_backtests(dataset_id)) as res:
            return res.json()["data"]

    @staticmethod
    def get_preset_by_id(id: int):
        with Req("get", URL.get_preset_by_id(id)) as res:
            return res.json()["data"]

    @staticmethod
    def get_all_presets_by_category(category: str):
        with Req("get", URL.get_presets_by_category(category)) as res:
            return res.json()["data"]

    @staticmethod
    def get_quant_stats_summary(backtest_id: int):
        with Req("get", URL.get_quant_stats_summary(backtest_id)) as res:
            return res


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
    def create_train_job(model_id: int, body):
        with Req("post", URL.create_train_job(model_id), json=body) as res:
            return res.json()["id"]

    @staticmethod
    def create_model_backtest(train_job_id, body):
        with Req("post", URL.create_model_backtest(train_job_id), json=body) as res:
            return res

    @staticmethod
    def create_manual_backtest(body):
        with Req("post", URL.create_manual_backtest(), json=body) as res:
            return res

    @staticmethod
    def create_long_short_backtest(body):
        with Req("post", URL.create_long_short_backtest(), json=body) as res:
            return res

    @staticmethod
    def create_code_preset(body):
        with Req("post", URL.create_code_preset(), json=body) as res:
            return res.json()["id"]

    @staticmethod
    def create_ml_based_backtest(body):
        with Req("post", URL.create_ml_based_backtest(), json=body) as res:
            return res

    @staticmethod
    def run_rule_based_sim_on_universe(body):
        with Req("post", URL.create_rule_based_sim_on_universe(), json=body) as res:
            return res


class Put:
    @staticmethod
    def update_dataset_name(dataset_name: str, body):
        with Req("put", URL.t_update_dataset_name(dataset_name), json=body) as res:
            return res.json()

    @staticmethod
    def update_timeseries_col(dataset_name: str, body):
        with Req("put", URL.update_timeseries_col(dataset_name), json=body) as res:
            return res.json()

    @staticmethod
    def update_target_col(dataset_name: str, target_col: str):
        with Req("put", URL.update_target_col(dataset_name, target_col)) as res:
            return res


class Delete:
    @staticmethod
    def datasets(list_of_dataset_names):
        with Req("delete", URL.t_get_tables(), json=list_of_dataset_names) as res:
            return res
