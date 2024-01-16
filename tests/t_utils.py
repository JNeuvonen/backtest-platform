import os
import pandas as pd
import requests
from pyserver.src.config import append_app_data_path

from tests.t_constants import URL, DatasetMetadata
from tests.t_context import t_file


def t_fetch_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def t_get_timeseries_col(table_item):
    return table_item["timeseries_col"]


def read_csv_to_df(path):
    return pd.read_csv(append_app_data_path(path))


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


class FetchData:
    @staticmethod
    def get_tables():
        return t_fetch_json(URL.t_get_tables())["tables"]
