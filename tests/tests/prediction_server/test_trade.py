import time
import pytest
from t_utils import Get, Post, Put
from fixtures.strategy import strategy_simple_1, update_strat_on_trade_open_test_case_1
from fixtures.trade import close_trade_body_test_case_1, trade_test_backend_sanity
from tests.prediction_server.fixtures.strategy_group import test_case_dump


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, create_api_key):
    res_strat = Post.create_strategy_group(create_api_key, body=test_case_dump)
    id = res_strat.text
    trade_id = Post.create_trade(create_api_key, trade_test_backend_sanity(int(1)))
    trades = Get.fetch_trades(create_api_key)
    assert len(trades) == 1
    time.sleep(10000)


@pytest.mark.acceptance
def test_closing_long_trade(cleanup_db, create_api_key):
    res_strat = Post.create_strategy(create_api_key, body=strategy_simple_1())
    strat_id = int(res_strat.text)
    res_trade = Post.create_trade(create_api_key, trade_test_backend_sanity(strat_id))
    trade_id = int(res_trade.text)
    Put.update_strategy(
        create_api_key, update_strat_on_trade_open_test_case_1(trade_id, strat_id)
    )
    Put.update_trade_close(
        create_api_key, int(strat_id), close_trade_body_test_case_1()
    )
