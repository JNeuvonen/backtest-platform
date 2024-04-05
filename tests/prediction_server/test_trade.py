import pytest
from t_utils import Get, Post
from fixtures.strategy import strategy_simple_1
from fixtures.trade import trade_test_backend_sanity


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, create_api_key):
    res_strat = Post.create_strategy(create_api_key, body=strategy_simple_1())
    id = res_strat.text
    Post.create_trade(create_api_key, trade_test_backend_sanity(int(id)))
    trades = Get.fetch_trades(create_api_key)
    assert len(trades) == 1
