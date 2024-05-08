import logging
import torch
from api_binance import non_async_save_historical_klines
from log import get_logger
from math_utils import safe_divide
from query_backtest import BacktestQuery
from query_backtest_history import BacktestHistoryQuery
from query_backtest_statistics import (
    BacktestStatisticsQuery,
)
from code_gen_template import BACKTEST_MODEL_TEMPLATE_V2
from db import ms_to_years
from query_model_columns import ModelColumnsQuery
from query_model_scalers import ModelScalerQuery
from backtest_utils import (
    START_BALANCE,
    MLBasedBacktest,
    calc_max_drawdown,
    calc_profit_factor,
    create_trade_entries,
    exec_load_model,
    get_cagr,
    get_trade_details,
    run_transformations_on_dataset,
)
from constants import BINANCE_BACKTEST_PRICE_COL, DomEventChannels, NullFillStrategy
from dataset import (
    df_fill_nulls_on_dataframe,
    drop_all_redundant_cols_and_reorder,
    read_dataset_to_mem,
    remove_inf_values,
    scale_df,
)
from query_data_transformation import DataTransformationQuery
from query_dataset import DatasetQuery
from query_model import ModelQuery
from query_trainjob import TrainJobQuery
from request_types import BodyMLBasedBacktest


def get_trading_decisions(prediction, enter_trade_cond, exit_trade_cond):
    code = BACKTEST_MODEL_TEMPLATE_V2

    replacements = {
        "{ENTER_TRADE_DECISION_FUNC}": enter_trade_cond,
        "{EXIT_TRADE_DECISION_FUNC}": exit_trade_cond,
        "{PREDICTION}": prediction,
    }

    for key, value in replacements.items():
        code = code.replace(key, str(value))

    results_dict = {"should_open_trade": None, "should_close_trade": None}
    exec(code, globals(), results_dict)
    return results_dict


async def run_ml_based_backtest(body: BodyMLBasedBacktest):
    dataset = DatasetQuery.fetch_dataset_by_name(body.dataset_name)
    data_transformations = DataTransformationQuery.get_transformations_by_dataset(
        dataset.id
    )

    if body.fetch_latest_data is True:
        non_async_save_historical_klines(dataset.symbol, dataset.interval, True)
        DatasetQuery.update_price_column(
            dataset.dataset_name, BINANCE_BACKTEST_PRICE_COL
        )
        run_transformations_on_dataset(data_transformations, dataset)

    dataset_df = read_dataset_to_mem(dataset.dataset_name)

    model_metadata = ModelQuery.fetch_model_by_id(body.id_of_model)
    trainjob = TrainJobQuery.get_train_job(body.train_run_id)

    model = exec_load_model(
        model_class_code=model_metadata.model_code,
        train_job_id=trainjob.id,
        epoch_nr=body.epoch,
        x_input_size=model_metadata.x_num_cols,
    )

    split_start_index = int(len(dataset_df) * body.backtest_data_range[0] / 100)
    split_end_index = int(len(dataset_df) * body.backtest_data_range[1] / 100)

    dataset_df = dataset_df.iloc[split_start_index:split_end_index]
    price_df = dataset_df[dataset.price_column].copy()
    kline_open_time_df = dataset_df[dataset.timeseries_column].copy()

    scaler = ModelScalerQuery.get_scaler_by_model_id(model_metadata.id)
    models_columns = ModelColumnsQuery.get_columns_by_model_id(model_metadata.id)

    if models_columns is None:
        raise Exception("Model columns was unexpectedly None")

    drop_all_redundant_cols_and_reorder(dataset_df, models_columns)
    df_fill_nulls_on_dataframe(
        dataset_df, NullFillStrategy(int(model_metadata.null_fill_strategy))
    )
    remove_inf_values(dataset_df)
    scale_df(dataset_df, scaler)

    first_price = price_df.iloc[0]

    backtest = MLBasedBacktest(body, dataset, first_price)

    for (_, row), price, kline_open_time in zip(
        dataset_df.iterrows(), price_df, kline_open_time_df
    ):
        tensor_row = torch.tensor(row.values, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(tensor_row).item()

        backtest.process_bar(
            kline_open_time,
            price,
            get_trading_decisions(
                prediction, body.enter_trade_cond, body.exit_trade_cond
            ),
        )

    profit_factor = calc_profit_factor(backtest.completed_trades)
    max_drawdown_dict = calc_max_drawdown(backtest.position.position_history)

    end_balance = backtest.position.net_value
    benchmark_end_balance = backtest.position.position_history[-1]["benchmark_price"]

    strategy_cagr = get_cagr(
        end_balance,
        START_BALANCE,
        ms_to_years(backtest.stats.cumulative_time),
    )

    benchmark_cagr = get_cagr(
        benchmark_end_balance,
        START_BALANCE,
        ms_to_years(backtest.stats.cumulative_time),
    )

    strat_time_exp = backtest.stats.position_held_time / backtest.stats.cumulative_time
    strat_risk_adjusted_cagr = safe_divide(strategy_cagr, strat_time_exp, 0)

    trade_count = len(backtest.completed_trades)
    trade_details_dict = get_trade_details(backtest.completed_trades)

    backtest_dict = {
        "name": body.name,
        "dataset_id": dataset.id,
        "candle_interval": dataset.interval,
        "ml_long_cond": body.enter_trade_cond,
        "ml_short_cond": body.exit_trade_cond,
        "use_time_based_close": body.use_time_based_close,
        "use_profit_based_close": body.use_profit_based_close,
        "use_stop_loss_based_close": body.use_stop_loss_based_close,
        "klines_until_close": body.klines_until_close,
        "backtest_range_start": body.backtest_data_range[0],
        "backtest_range_end": body.backtest_data_range[1],
    }

    backtest_id = BacktestQuery.create_entry(backtest_dict)

    backtest_statistics_dict = {
        "backtest_id": backtest_id,
        "profit_factor": profit_factor,
        "start_balance": START_BALANCE,
        "end_balance": end_balance,
        "result_perc": (end_balance / START_BALANCE - 1) * 100,
        "take_profit_threshold_perc": body.take_profit_threshold_perc,
        "stop_loss_threshold_perc": body.stop_loss_threshold_perc,
        "best_trade_result_perc": trade_details_dict["best_trade_result_perc"],
        "worst_trade_result_perc": trade_details_dict["worst_trade_result_perc"],
        "mean_hold_time_sec": trade_details_dict["mean_hold_time_sec"],
        "mean_return_perc": trade_details_dict["mean_return_perc"],
        "buy_and_hold_result_net": benchmark_end_balance - START_BALANCE,
        "buy_and_hold_result_perc": (benchmark_end_balance / START_BALANCE - 1) * 100,
        "share_of_winning_trades_perc": trade_details_dict[
            "share_of_winning_trades_perc"
        ],
        "share_of_losing_trades_perc": trade_details_dict[
            "share_of_losing_trades_perc"
        ],
        "max_drawdown_perc": max_drawdown_dict["drawdown_strategy_perc"],
        "benchmark_drawdown_perc": max_drawdown_dict["drawdown_benchmark_perc"],
        "cagr": strategy_cagr,
        "market_exposure_time": strat_time_exp,
        "risk_adjusted_return": strat_risk_adjusted_cagr,
        "buy_and_hold_cagr": benchmark_cagr,
        "trade_count": trade_count,
        "slippage_perc": body.slippage_perc,
        "short_fee_hourly": body.short_fee_hourly,
        "trading_fees_perc": body.trading_fees_perc,
    }

    BacktestStatisticsQuery.create_entry(backtest_statistics_dict)

    for item in backtest.position.position_history:
        item["kline_open_time"] = int(item["kline_open_time"] / 1000)

    BacktestHistoryQuery.create_many(backtest_id, backtest.position.position_history)
    create_trade_entries(backtest_id, backtest.completed_trades)

    logger = get_logger()
    logger.log(
        f"Finished ML based backtest. Profit: {(end_balance/START_BALANCE - 1) * 100:.2f}%",
        logging.INFO,
        True,
        True,
        DomEventChannels.REFETCH_COMPONENT.value,
    )
