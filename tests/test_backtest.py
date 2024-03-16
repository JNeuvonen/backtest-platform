import pytest

from tests.fixtures import (
    close_long_trade_cond_basic,
    close_short_trade_cond_basic,
    create_manual_backtest,
    open_long_trade_cond_basic,
    open_short_trade_cond_basic,
)
from tests.t_utils import Fetch, Post


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, fixt_btc_small_1h):
    dataset = Fetch.get_dataset_by_name(fixt_btc_small_1h.name)
    backtest_body = create_manual_backtest(
        dataset["id"],
        True,
        open_long_trade_cond_basic(),
        open_short_trade_cond_basic(),
        close_long_trade_cond_basic(),
        close_short_trade_cond_basic(),
    )

    Post.create_manual_backtest(backtest_body)
