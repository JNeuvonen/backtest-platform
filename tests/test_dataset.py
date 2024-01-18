from typing import List
import pytest
import sys
import requests

from tests.t_conf import SERVER_SOURCE_DIR
from tests.t_constants import BinanceCols, BinanceData, DatasetMetadata
from tests.t_populate import t_upload_dataset
from tests.t_utils import (
    Fetch,
    Post,
    Put,
    PythonCode,
    add_object_to_add_cols_payload,
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
def test_route_update_dataset_name(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    _ = cleanup_db
    NEW_DATASET_NAME = "test_name_123"

    Put.update_dataset_name(
        fixt_btc_small_1h.name, body={"new_dataset_name": NEW_DATASET_NAME}
    )
    Fetch.get_dataset_by_name(NEW_DATASET_NAME)


@pytest.mark.acceptance
def test_route_get_dataset_col_info(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    _ = cleanup_db

    res = Fetch.get_dataset_col_info(fixt_btc_small_1h.name, BinanceCols.LOW_PRICE)
    dataset = res[0]
    assert len(dataset["rows"]) == len(dataset["kline_open_time"])


@pytest.mark.acceptance
def test_route_update_timeseries_col(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    _ = cleanup_db
    Put.update_timeseries_col(
        fixt_btc_small_1h.name, {"new_timeseries_col": BinanceCols.LOW_PRICE}
    )
    dataset = Fetch.get_dataset_by_name(fixt_btc_small_1h.name)
    timeseries_col = dataset["timeseries_col"]
    assert timeseries_col == BinanceCols.LOW_PRICE


@pytest.mark.acceptance
def test_route_dataset_add_cols(fixt_add_all_downloaded_datasets):
    BTC_1MO = BinanceData.BTCUSDT_1MO
    AAVE_1MO = BinanceData.AAVEUSDT_1MO
    SUSHI_1MO = BinanceData.SUSHIUSDT_1MO

    sushi_dataset = Fetch.get_dataset_by_name(SUSHI_1MO.name)
    btc_dataset_before_add = Fetch.get_dataset_by_name(BTC_1MO.name)

    payload = []

    aave_added_cols = [
        BinanceCols.HIGH_PRICE,
        BinanceCols.OPEN_PRICE,
    ]
    add_object_to_add_cols_payload(payload, AAVE_1MO.name, aave_added_cols)

    sushi_added_cols = [
        BinanceCols.TAKER_BUY_BASE_ASSET_VOLUME,
        BinanceCols.TAKER_BUY_QUOTE_ASSET_VOLUME,
    ]
    add_object_to_add_cols_payload(payload, SUSHI_1MO.name, sushi_added_cols)

    Post.add_columns(BTC_1MO.name, body=payload)

    btc_dataset_after_add = Fetch.get_dataset_by_name(BTC_1MO.name)

    assert len(btc_dataset_before_add["columns"]) + len(aave_added_cols) + len(
        sushi_added_cols
    ) == len(btc_dataset_after_add["columns"])

    null_counts = btc_dataset_after_add["null_counts"]

    for key, value in null_counts.items():
        if SUSHI_1MO.name in key:
            # assert no extra nulls
            assert (
                value
                == btc_dataset_before_add["row_count"] - sushi_dataset["row_count"]
            )


@pytest.mark.acceptance
def test_route_dataset_add_cols_fill_closest(fixt_add_all_downloaded_datasets):
    BTC_1MO = BinanceData.BTCUSDT_1MO
    AAVE_1MO = BinanceData.AAVEUSDT_1MO
    SUSHI_1MO = BinanceData.SUSHIUSDT_1MO

    payload = []

    aave_added_cols = [
        BinanceCols.HIGH_PRICE,
        BinanceCols.OPEN_PRICE,
    ]
    add_object_to_add_cols_payload(payload, AAVE_1MO.name, aave_added_cols)

    sushi_added_cols = [
        BinanceCols.TAKER_BUY_BASE_ASSET_VOLUME,
        BinanceCols.TAKER_BUY_QUOTE_ASSET_VOLUME,
    ]
    add_object_to_add_cols_payload(payload, SUSHI_1MO.name, sushi_added_cols)

    Post.add_columns(BTC_1MO.name, body=payload, null_fill_strategy="CLOSEST")

    btc_dataset_after_add = Fetch.get_dataset_by_name(BTC_1MO.name)
    null_counts = btc_dataset_after_add["null_counts"]

    for _, value in null_counts.items():
        assert value == 0


@pytest.mark.acceptance
def test_route_exec_python(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    res_open_price = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.OPEN_PRICE
    )
    res_quote_asset_vol = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.QUOTE_ASSET_VOLUME
    )
    assert res_open_price != res_quote_asset_vol

    python_program = PythonCode.append_code(
        fixt_btc_small_1h.name,
        f'dataset["{BinanceCols.OPEN_PRICE}"] = dataset["{BinanceCols.QUOTE_ASSET_VOLUME}"]',
    )

    Post.exec_python(body={"code": python_program})
    res_open_price = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.OPEN_PRICE
    )
    res_quote_asset_vol = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.QUOTE_ASSET_VOLUME
    )
    assert res_open_price == res_quote_asset_vol


@pytest.mark.acceptance
def test_route_exec_python_multiply_col(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    res_open_price_before = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.OPEN_PRICE
    )
    python_program = PythonCode.append_code(
        fixt_btc_small_1h.name,
        f'dataset["{BinanceCols.OPEN_PRICE}"] = dataset["{BinanceCols.OPEN_PRICE}"] * 3',
    )

    Post.exec_python(body={"code": python_program})

    res_open_price_after = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.OPEN_PRICE
    )

    for i in range(len(res_open_price_before[0]["rows"])):
        before = res_open_price_before[0]["rows"][i][0]
        after = res_open_price_after[0]["rows"][i][0]
        assert before * 3 == after
