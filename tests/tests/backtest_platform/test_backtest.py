import pytest
import time
from constants import NullFillStrategy
from tests.backtest_platform.conftest import fixt_add_many_datasets
from tests.backtest_platform.t_utils import (
    create_model_body,
    gen_cdl3_star_in_south_transformation,
    gen_data_transformations,
    gen_ml_based_test_data_transformations,
)

from tests.backtest_platform.fixtures import (
    close_long_trade_cond_basic,
    create_manual_backtest,
    create_train_job_basic,
    criterion_basic,
    linear_model_basic,
    long_short_buy_cond_basic,
    long_short_pair_exit_code_basic,
    long_short_sell_cond_basic,
    ml_based_enter_long_cond_basic,
    ml_based_enter_short_cond_basic,
    ml_based_exit_long_cond_basic,
    ml_based_exit_short_cond_basic,
    open_long_trade_cond_basic,
    body_rule_based_on_universe_v1,
    body_rule_based_on_universe_debug_crash,
)
from tests.backtest_platform.t_utils import Fetch, Post
from tests.backtest_platform.fixtures import (
    backtest_rule_based_v2,
    backtest_rule_based_debug_scalar_divide,
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


@pytest.mark.long_short_test
def test_long_short_backtest(cleanup_db, fixt_add_blue_chip_1d_datasets):
    datasets = fixt_add_blue_chip_1d_datasets
    body = backtest_rule_based_v2
    dataset_names = []

    first_dataset = datasets[0]
    data_transformation_ids = gen_data_transformations(first_dataset.name)

    for item in datasets:
        dataset_names.append(item.pair_name)

    body["datasets"] = dataset_names
    body["data_transformations"] = data_transformation_ids
    body["buy_cond"] = long_short_buy_cond_basic()
    body["sell_cond"] = long_short_sell_cond_basic()
    body["exit_cond"] = long_short_pair_exit_code_basic()
    body["max_simultaneous_positions"] = 15
    body["max_leverage_ratio"] = 2.5
    body["candle_interval"] = "1d"
    body["fetch_latest_data"] = False

    Post.create_long_short_backtest(body)
    time.sleep(1000)


@pytest.mark.acceptance
def test_ml_based_backtest(fixt_add_dataset_for_ml_based_backtest):
    ML_MODEL_NAME = "Example model"

    gen_ml_based_test_data_transformations(
        fixt_add_dataset_for_ml_based_backtest[0].name
    )

    body = create_model_body(
        name=ML_MODEL_NAME,
        drop_cols=[],
        null_fill_strategy=NullFillStrategy.NONE.value,
        model=linear_model_basic(),
        hyper_params_and_optimizer_code=criterion_basic(),
        validation_split=[70, 100],
    )

    model_id = Post.create_model(fixt_add_dataset_for_ml_based_backtest[0].name, body)
    train_job_id = Post.create_train_job(model_id, body=create_train_job_basic())

    body = backtest_rule_based_v2

    body["candle_interval"] = "1d"
    body["dataset_name"] = fixt_add_dataset_for_ml_based_backtest[0].name
    body["fetch_latest_data"] = True
    body["enter_long_trade_cond"] = ml_based_enter_long_cond_basic()
    body["exit_long_trade_cond"] = ml_based_exit_long_cond_basic()
    body["enter_short_trade_cond"] = ml_based_enter_short_cond_basic()
    body["exit_short_trade_cond"] = ml_based_exit_short_cond_basic()
    body["allow_shorts"] = True
    body["train_run_id"] = train_job_id
    body["id_of_model"] = model_id
    body["epoch"] = 4

    Post.create_ml_based_backtest(body=body)


@pytest.mark.dev
def test_rule_based_on_universe(cleanup_db, fixt_add_blue_chip_1d_datasets):
    datasets = fixt_add_blue_chip_1d_datasets
    body = body_rule_based_on_universe_v1

    first_dataset = datasets[0]
    data_transformation_ids = gen_data_transformations(first_dataset.name)
    body = body_rule_based_on_universe_v1
    body["data_transformations"] = data_transformation_ids
    dataset_names = []
    for item in datasets:
        dataset_names.append(item.pair_name)
    body["datasets"] = dataset_names

    Post.run_rule_based_sim_on_universe(body=body)


@pytest.mark.dev
def test_rule_based_on_universe_issue_with_scalar_div(
    cleanup_db, fixt_add_blue_chip_1d_datasets
):
    datasets = fixt_add_blue_chip_1d_datasets
    body = backtest_rule_based_debug_scalar_divide

    first_dataset = datasets[0]
    data_transformation_ids = gen_data_transformations(first_dataset.name)
    body["data_transformations"] = data_transformation_ids
    Post.run_rule_based_sim_on_universe(body=body)


@pytest.mark.debug
def test_rule_based_on_universe_issue_with_scalar_div(
    cleanup_db, fixt_add_blue_chip_1d_datasets
):
    datasets = fixt_add_blue_chip_1d_datasets
    body = body_rule_based_on_universe_debug_crash

    first_dataset = datasets[0]
    data_transformation_ids = gen_cdl3_star_in_south_transformation(first_dataset.name)
    body["data_transformations"] = data_transformation_ids
    Post.run_rule_based_sim_on_universe(body=body)
