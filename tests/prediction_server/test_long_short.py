import time
import pytest

from tests.prediction_server.fixtures.long_short import (
    create_binance_trade_response,
    long_short_body_basic,
    long_short_test_enter_trade,
)
from tests.prediction_server.t_utils import Post


@pytest.mark.long_short_dev
def test_setup_sanity(cleanup_db, fixt_create_master_acc):
    longshort_body = long_short_body_basic()

    Post.create_long_short_strategy(fixt_create_master_acc, longshort_body)

    time.sleep(1000000)


@pytest.mark.acceptance
def test_longshort_enter_trade(cleanup_db, fixt_create_master_acc):
    # longshort_body = long_short_body_basic()
    # Post.create_long_short_strategy(fixt_create_master_acc, longshort_body)
    # time.sleep(20)
    # longshort_enter_trade_body = long_short_test_enter_trade()
    # Post.enter_longshort_trade(fixt_create_master_acc, 1, longshort_enter_trade_body)
    pass


@pytest.mark.acceptance
def test_longshort_exit_trade(cleanup_db, fixt_create_master_acc):
    order_res_1 = create_binance_trade_response(
        symbol="BTCUSDT",
        orderId=12345,
        clientOrderId="testOrder123",
        transactTime=1622547801000,
        price="50000.00",
        origQty="0.002",
        executedQty="0.002",
        cummulativeQuoteQty="100.00",
        status="FILLED",
        timeInForce="GTC",
        type="LIMIT",
        side="BUY",
        marginBuyBorrowAmount=50.0,
        marginBuyBorrowAsset="USDT",
        isIsolated=True,
    )

    order_res_2 = create_binance_trade_response(
        symbol="ETHUSDT",
        orderId=12346,
        clientOrderId="testOrder124",
        transactTime=1622547901000,
        price="2500.00",
        origQty="0.1",
        executedQty="0.1",
        cummulativeQuoteQty="250.00",
        status="PARTIALLY_FILLED",
        timeInForce="IOC",
        type="MARKET",
        side="SELL",
        marginBuyBorrowAmount=25.0,
        marginBuyBorrowAsset="USDT",
        isIsolated=False,
    )

    Post.exit_longshort_trade(
        fixt_create_master_acc,
        3,
        {"long_side_order": order_res_1, "short_side_order": order_res_2},
    )
