from typing import List
import pytest
import sys

from tests.t_conf import SERVER_SOURCE_DIR
from tests.t_constants import BinanceCols, BinanceData, DatasetMetadata
from tests.t_populate import t_upload_dataset
from tests.t_utils import (
    Fetch,
    Post,
    t_get_timeseries_col,
)

sys.path.append(SERVER_SOURCE_DIR)


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, fixt_btc_small_1h):
    _ = cleanup_db
    dataset = fixt_btc_small_1h
    tables = Fetch.get_tables()
    assert len(tables) == 1
    assert t_get_timeseries_col(tables[0]) == dataset.timeseries_col


@pytest.mark.acceptance
def test_upload_timeseries_data(cleanup_db):
    _ = cleanup_db
    response = t_upload_dataset(BinanceData.BTCUSDT_1H_2023_06)
    assert response.status_code == 200
    tables = Fetch.get_tables()
    assert len(tables) == 1
    assert (
        t_get_timeseries_col(tables[0]) == BinanceData.BTCUSDT_1H_2023_06.timeseries_col
    )


@pytest.mark.acceptance
def test_route_rename_column(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    _ = cleanup_db
    dataset = fixt_btc_small_1h

    TEST_COL_NAME = "new_open_price"
    Post.rename_column(
        fixt_btc_small_1h.name,
        {
            "old_col_name": BinanceCols.OPEN_PRICE,
            "new_col_name": TEST_COL_NAME,
        },
    )
    dataset_from_db = Fetch.get_dataset_by_name(dataset_name=dataset.name)
    dataset_cols = dataset_from_db["columns"]

    assert TEST_COL_NAME in dataset_cols
    assert BinanceCols.OPEN_PRICE not in dataset_cols

    TEST_COL_NAME = "new_kline_open_time"
    Post.rename_column(
        fixt_btc_small_1h.name,
        {
            "old_col_name": BinanceCols.KLINE_OPEN_TIME,
            "new_col_name": TEST_COL_NAME,
        },
    )

    dataset_from_db = Fetch.get_dataset_by_name(dataset_name=dataset.name)
    dataset_cols = dataset_from_db["columns"]

    assert TEST_COL_NAME in dataset_cols
    assert BinanceCols.KLINE_OPEN_TIME not in dataset_cols


@pytest.mark.acceptance
def test_route_all_columns(cleanup_db, fixt_add_many_datasets: List[DatasetMetadata]):
    _ = cleanup_db
    table_col_map = Fetch.get_all_tables_and_columns()
    assert len(table_col_map) == len(fixt_add_many_datasets)


@pytest.mark.acceptance
def test_route_update_dataset_name(cleanup_db, fixt_btc_small_1h):
    _ = cleanup_db
