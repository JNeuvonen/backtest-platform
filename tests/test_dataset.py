import pytest
import sys

from tests.conftest import (
    t_read_binance_df,
)
from tests.t_conf import SERVER_SOURCE_DIR
from tests.t_constants import BinanceData
from tests.t_populate import t_upload_dataset
from tests.t_utils import FetchData, t_get_timeseries_col

sys.path.append(SERVER_SOURCE_DIR)
from utils import add_to_datasets_db


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db):
    _ = cleanup_db
    df = t_read_binance_df(BinanceData.BTCUSDT_1H_2023_06.path)
    add_to_datasets_db(df, BinanceData.BTCUSDT_1H_2023_06.name)
    tables = FetchData.get_tables()
    assert len(tables) == 1


@pytest.mark.acceptance
def test_upload_timeseries_data(cleanup_db):
    _ = cleanup_db
    dataset = BinanceData.BTCUSDT_1H_2023_06
    response = t_upload_dataset(dataset)
    assert response.status_code == 200
    tables = FetchData.get_tables()
    assert len(tables) == 1
    assert t_get_timeseries_col(tables[0]) == dataset.timeseries_col
