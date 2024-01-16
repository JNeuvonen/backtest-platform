import json
import pytest
import sys

import requests
from tests.conftest import (
    binance_path_to_dataset_name,
    read_binance_df,
)
from tests.t_constants import BinanceData, FetchData
from tests.t_context import binance_file

sys.path.append("pyserver/src")
from utils import add_to_datasets_db


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db):
    _ = cleanup_db
    df = read_binance_df(BinanceData.BTCUSDT_1H_2023_06)
    add_to_datasets_db(df, binance_path_to_dataset_name(BinanceData.BTCUSDT_1H_2023_06))
    tables = FetchData.get_tables()
    assert len(tables) == 1


@pytest.mark.acceptance
def test_upload_timeseries_data(cleanup_db):
    _ = cleanup_db
    url = "http://localhost:8000/dataset/upload-timeseries-data?dataset_name=btcusdt_1h"
    with binance_file(BinanceData.BTCUSDT_1H_2023_06) as file:
        response = requests.post(
            url,
            files={"file": ("file.csv", file)},
        )
    result = response.json()

    tables = FetchData.get_tables()
    assert len(tables) == 1
    print("Hello world")
