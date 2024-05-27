import time
import pytest
from tests.prediction_server.fixtures.binance_mock import ORDER_RES_1, ORDER_RES_2

from tests.prediction_server.fixtures.long_short import (
    create_binance_trade_response,
    long_short_body_basic,
    long_short_test_enter_trade,
)
from tests.prediction_server.t_utils import Post
from common_python.pred_serv_models.longshortpair import LongShortPairQuery


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, fixt_create_master_acc):
    longshort_body = long_short_body_basic()
    id = Post.create_long_short_strategy(fixt_create_master_acc, longshort_body)
    print(id)


@pytest.mark.acceptance
def test_longshort_enter_trade(cleanup_db, fixt_create_master_acc):
    longshort_body = long_short_body_basic()
    Post.create_long_short_strategy(fixt_create_master_acc, longshort_body)

    LongShortPairQuery.create_entry(
        {
            "long_short_group_id": 1,
            "buy_ticker_id": 1,
            "sell_ticker_id": 2,
            "buy_ticker_dataset_name": "DUSKUSDT_TEST_LONG_SHORT",
            "sell_ticker_dataset_name": "TRXUSDT_TEST_LONG_SHORT",
            "buy_symbol": "TRXUSDT",
            "buy_base_asset": "TRX",
            "buy_quote_asset": "USDT",
            "sell_symbol": "DUSKUSDT",
            "sell_base_asset": "DUSK",
            "sell_quote_asset": "USDT",
            "buy_qty_precision": 2,
            "sell_qty_precision": 2,
        }
    )

    body = {}
    body["long_side_order"] = ORDER_RES_1
    body["short_side_order"] = ORDER_RES_2
    enter_res = Post.enter_longshort_trade(fixt_create_master_acc, 1, body)
    print("test")


@pytest.mark.acceptance
def test_longshort_exit_trade(cleanup_db, fixt_create_master_acc):
    longshort_body = long_short_body_basic()
    Post.create_long_short_strategy(fixt_create_master_acc, longshort_body)
    LongShortPairQuery.create_entry(
        {
            "long_short_group_id": 1,
            "buy_ticker_id": 1,
            "sell_ticker_id": 2,
            "buy_ticker_dataset_name": "DUSKUSDT_TEST_LONG_SHORT",
            "sell_ticker_dataset_name": "TRXUSDT_TEST_LONG_SHORT",
            "buy_symbol": "TRXUSDT",
            "buy_base_asset": "TRX",
            "buy_quote_asset": "USDT",
            "sell_symbol": "DUSKUSDT",
            "sell_base_asset": "DUSK",
            "sell_quote_asset": "USDT",
            "buy_qty_precision": 2,
            "sell_qty_precision": 2,
        }
    )
    body = {}
    body["long_side_order"] = ORDER_RES_1
    body["short_side_order"] = ORDER_RES_2
    enter_res = Post.enter_longshort_trade(fixt_create_master_acc, 1, body)
    Post.exit_longshort_trade(
        fixt_create_master_acc,
        1,
        {"long_side_order": ORDER_RES_1, "short_side_order": ORDER_RES_2},
    )
