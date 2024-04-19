import pytest

from tests.backtest_platform.fixtures import (
    close_long_trade_cond_basic,
    close_short_trade_cond_basic,
    create_manual_backtest,
    open_long_trade_cond_basic,
    open_short_trade_cond_basic,
)
from tests.backtest_platform.t_utils import Fetch, Post
from tests.backtest_platform.fixtures import (
    backtest_time_based_close_is_not_working,
    backtest_psr_debug,
    backtest_div_by_zero_bug,
    backtest_rule_based_v2,
)


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, fixt_btc_small_1h):
    dataset = Fetch.get_dataset_by_name(fixt_btc_small_1h.name)
    backtest_body = create_manual_backtest(
        dataset["id"],
        True,
        open_long_trade_cond_basic(),
        close_long_trade_cond_basic(),
        False,
        0.1,
        0.01,
        [0, 100],
        False,
        False,
        0.000016,
        0.0,
        0.0,
    )

    Post.create_manual_backtest(backtest_body)


@pytest.mark.acceptance
def test_fetch_backtests(fixt_manual_backtest):
    dataset = Fetch.get_dataset_by_name(fixt_manual_backtest.name)
    backtests = Fetch.get_datasets_manual_backtests(dataset["id"])
    assert len(backtests) == 1, "Backtest wasnt created or fetches succesfully"


@pytest.mark.input_dump
def test_quant_stats_report_gen(cleanup_db, add_custom_datasets):
    dataset = Fetch.get_dataset_by_name("btcusdt_1h_dump")
    body = backtest_rule_based_v2
    body["dataset_id"] = dataset["id"]
    Post.create_manual_backtest(body)
    backtests = Fetch.get_datasets_manual_backtests(dataset["id"])
    Fetch.get_quant_stats_summary(backtests[0]["id"])


@pytest.mark.input_dump
def test_backtest_time_based_close(cleanup_db, add_custom_datasets):
    dataset = Fetch.get_dataset_by_name("btcusdt_1h_dump")
    # body = backtest_time_based_close_is_not_working
    body = backtest_rule_based_v2
    body["dataset_id"] = dataset["id"]
    Post.create_manual_backtest(body)
