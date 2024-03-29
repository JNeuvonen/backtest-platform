from typing import List
import pytest
import sys

from decimal import Decimal
from tests.backtest_platform.fixtures import criterion_basic, linear_model_basic
from tests.backtest_platform.t_conf import SERVER_SOURCE_DIR
from tests.backtest_platform.t_constants import (
    BinanceCols,
    BinanceData,
    DatasetMetadata,
)
from tests.backtest_platform.t_populate import t_upload_dataset
from tests.backtest_platform.t_utils import (
    Delete,
    Fetch,
    Post,
    Put,
    add_object_to_add_cols_payload,
    create_model_body,
    t_get_timeseries_col,
)

sys.path.append(SERVER_SOURCE_DIR)

from utils import PythonCode
from constants import NullFillStrategy


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
def test_route_update_target_col(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    _ = cleanup_db
    Put.update_target_col(fixt_btc_small_1h.name, BinanceCols.LOW_PRICE)
    dataset = Fetch.get_dataset_by_name(fixt_btc_small_1h.name)
    target_col = dataset["target_col"]
    assert target_col == BinanceCols.LOW_PRICE


@pytest.mark.acceptance
def test_route_dataset_add_cols(cleanup_db, fixt_add_all_downloaded_datasets):
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
def test_route_dataset_add_cols_fill_closest(
    cleanup_db, fixt_add_all_downloaded_datasets
):
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

    Post.exec_python_on_col(
        fixt_btc_small_1h.name,
        BinanceCols.OPEN_PRICE,
        body={
            "code": f'dataset["{BinanceCols.OPEN_PRICE}"] = dataset["{BinanceCols.QUOTE_ASSET_VOLUME}"]\n',
        },
    )
    res_open_price = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.OPEN_PRICE
    )
    res_quote_asset_vol = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.QUOTE_ASSET_VOLUME
    )
    assert res_open_price[0]["rows"] == res_quote_asset_vol[0]["rows"]


@pytest.mark.acceptance
def test_route_exec_python_multiply_col(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    res_open_price_before = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.OPEN_PRICE
    )

    Post.exec_python_on_col(
        fixt_btc_small_1h.name,
        BinanceCols.OPEN_PRICE,
        body={
            "code": f"dataset[{PythonCode.COLUMN_SYMBOL}] = dataset[{PythonCode.COLUMN_SYMBOL}] * 3\n",
        },
    )

    res_open_price_after = Fetch.get_dataset_col_info(
        fixt_btc_small_1h.name, BinanceCols.OPEN_PRICE
    )

    for i in range(len(res_open_price_before[0]["rows"])):
        before = res_open_price_before[0]["rows"][i]
        after = res_open_price_after[0]["rows"][i]
        assert before * 3 == after


@pytest.mark.acceptance
def test_route_exec_python_on_dataset(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    dataset_before = Fetch.get_dataset_by_name(fixt_btc_small_1h.name)

    Post.exec_python_on_dataset(
        fixt_btc_small_1h.name,
        body={
            "code": f"for item in dataset.columns:\n{PythonCode.INDENT}dataset[item] = dataset[item] * 3\n",
        },
    )

    dataset_after = Fetch.get_dataset_by_name(fixt_btc_small_1h.name)

    idx = 0
    for head_before in dataset_before["head"]:
        before_list_mul_3 = [Decimal(x) * Decimal("3") for x in head_before]
        after_list = dataset_after["head"][idx]

        for i in range(len(before_list_mul_3)):
            # prevent rounding errors from failing tests
            assert (
                before_list_mul_3[i] <= 1.005 * after_list[i]
                or before_list_mul_3[i] >= 0.995 * after_list[i]
            )

    for tail_before in dataset_before["tail"]:
        before_list_mul_3 = [Decimal(x) * Decimal("3") for x in tail_before]
        after_list = dataset_after["tail"][idx]

        for i in range(len(before_list_mul_3)):
            # prevent rounding errors from failing tests
            assert (
                before_list_mul_3[i] <= 1.005 * after_list[i]
                or before_list_mul_3[i] >= 0.995 * after_list[i]
            )


@pytest.mark.acceptance
def test_route_create_model(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    body = create_model_body(
        name="Example model",
        drop_cols=[],
        null_fill_strategy=NullFillStrategy.CLOSEST.value,
        model=linear_model_basic(),
        hyper_params_and_optimizer_code=criterion_basic(),
        validation_split=[70, 100],
    )

    Post.create_model(fixt_btc_small_1h.name, body)


@pytest.mark.acceptance
def test_route_fetch_models(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    body = create_model_body(
        name="Example model",
        drop_cols=[],
        null_fill_strategy=NullFillStrategy.CLOSEST.value,
        model=linear_model_basic(),
        hyper_params_and_optimizer_code=criterion_basic(),
        validation_split=[70, 100],
    )
    Post.create_model(fixt_btc_small_1h.name, body)
    models = Fetch.get_dataset_models(fixt_btc_small_1h.name)
    assert len(models) == 1


@pytest.mark.acceptance
def test_dataset_pagination(cleanup_db, fixt_btc_small_1h: DatasetMetadata):
    PAGE_SIZE = 20
    res_valid = Fetch.get_dataset_pagination(fixt_btc_small_1h.name, 1, PAGE_SIZE)
    res_invalid = Fetch.get_dataset_pagination(fixt_btc_small_1h.name, 1000, 202000)

    assert len(res_valid) == PAGE_SIZE
    assert len(res_invalid) == 0


@pytest.mark.acceptance
def test_delete_datasets(cleanup_db, fixt_add_all_downloaded_datasets):
    datasets = [item.name for item in fixt_add_all_downloaded_datasets]
    Delete.datasets({"dataset_names": datasets})
