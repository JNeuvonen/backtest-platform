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
        False,
        0.1,
        0.01,
    )

    Post.create_manual_backtest(backtest_body)


@pytest.mark.acceptance
def test_fetch_backtests(fixt_manual_backtest):
    dataset = Fetch.get_dataset_by_name(fixt_manual_backtest.name)
    backtests = Fetch.get_datasets_manual_backtests(dataset["id"])
    assert len(backtests) == 1, "Backtest wasnt created or fetches succesfully"
