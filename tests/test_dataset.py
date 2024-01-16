import pytest
import sys

from tests.t_conf import SERVER_SOURCE_DIR
from tests.t_constants import BinanceData
from tests.t_populate import t_upload_dataset
from tests.t_utils import (
    FetchData,
    t_get_timeseries_col,
)

sys.path.append(SERVER_SOURCE_DIR)


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, fixt_btc_small_1h):
    _ = cleanup_db
    dataset = fixt_btc_small_1h
    tables = FetchData.get_tables()
    assert len(tables) == 1
    assert t_get_timeseries_col(tables[0]) == dataset.timeseries_col


@pytest.mark.acceptance
def test_upload_timeseries_data(cleanup_db):
    _ = cleanup_db
    response = t_upload_dataset(BinanceData.BTCUSDT_1H_2023_06)
    assert response.status_code == 200
    tables = FetchData.get_tables()
    assert len(tables) == 1
    assert (
        t_get_timeseries_col(tables[0]) == BinanceData.BTCUSDT_1H_2023_06.timeseries_col
    )
