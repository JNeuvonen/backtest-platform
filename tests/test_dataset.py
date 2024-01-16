import pytest
import sys

import requests
from tests.conftest import (
    read_binance_df,
)
from tests.t_constants import URL, BinanceData, FetchData
from tests.t_context import binance_file

sys.path.append("pyserver/src")
from utils import add_to_datasets_db


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db):
    _ = cleanup_db
    df = read_binance_df(BinanceData.BTCUSDT_1H_2023_06.path)
    add_to_datasets_db(df, BinanceData.BTCUSDT_1H_2023_06.name)
    tables = FetchData.get_tables()
    assert len(tables) == 1


@pytest.mark.acceptance
def test_upload_timeseries_data(cleanup_db):
    _ = cleanup_db
    with binance_file(BinanceData.BTCUSDT_1H_2023_06.path) as file:
        response = requests.post(
            URL.get_upload_dataset_url("btcusdt_1h"),
            files={"file": ("file.csv", file)},
        )

    assert response.status_code == 200
    tables = FetchData.get_tables()
    assert len(tables) == 1
