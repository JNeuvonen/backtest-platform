import pytest
import time
from tests.backtest_platform.t_utils import gen_data_transformations

from tests.backtest_platform.fixtures import (
    close_long_trade_cond_basic,
    create_manual_backtest,
    long_short_buy_cond_basic,
    long_short_pair_exit_code_basic,
    long_short_sell_cond_basic,
    open_long_trade_cond_basic,
)
from tests.backtest_platform.t_utils import Fetch, Post
from tests.backtest_platform.fixtures import (
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


@pytest.mark.dev
def test_long_short_backtest(fixt_add_many_datasets):
    body = backtest_rule_based_v2
    dataset_ids = []
    time.sleep(3)

    first_dataset = fixt_add_many_datasets[0]
    data_transformation_ids = gen_data_transformations(first_dataset.name)

    for item in fixt_add_many_datasets:
        dataset = Fetch.get_dataset_by_name(item.name)
        dataset_ids.append(dataset["id"])

    body["datasets"] = dataset_ids
    body["data_transformations"] = data_transformation_ids
    body["buy_cond"] = long_short_buy_cond_basic()
    body["sell_cond"] = long_short_sell_cond_basic()
    body["exit_cond"] = long_short_pair_exit_code_basic()

    Post.create_long_short_backtest(body)
