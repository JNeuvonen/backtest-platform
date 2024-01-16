import pytest
import sys

from tests.t_conf import SERVER_SOURCE_DIR
from tests.t_constants import BinanceData
from tests.t_populate import t_upload_dataset
from tests.t_utils import (
    FetchData,
    read_csv_to_df,
    t_get_timeseries_col,
)

sys.path.append(SERVER_SOURCE_DIR)
from utils import add_to_datasets_db


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db):
    _ = cleanup_db
    df = read_csv_to_df(BinanceData.BTCUSDT_1H_2023_06.path)
    add_to_datasets_db(df, BinanceData.BTCUSDT_1H_2023_06.name)
    tables = FetchData.get_tables()
    assert len(tables) == 1


@pytest.mark.acceptance
def test_upload_timeseries_data(cleanup_db, fixt_btc_small_1h):
    _ = cleanup_db
    response, dataset = fixt_btc_small_1h
    assert response.status_code == 200
    tables = FetchData.get_tables()
    assert len(tables) == 1
    assert t_get_timeseries_col(tables[0]) == dataset.timeseries_col


@pytest.mark.acceptance
def test_upload_timeseries_big_stream(cleanup_db, init_large_csv):
    _ = cleanup_db
    response = t_upload_dataset(init_large_csv)
    assert response.status_code == 200
    tables = FetchData.get_tables()
    assert len(tables) == 1
    assert t_get_timeseries_col(tables[0]) == init_large_csv.timeseries_col
    print("Hello world")
