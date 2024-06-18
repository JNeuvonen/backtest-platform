import multiprocessing
from typing import Set
from common_python.date import (
    format_ms_to_human_readable,
    get_diff_in_ms,
    get_time_elapsed,
)
from common_python.math import get_cagr, ms_to_years
from common_python.trading.utils import (
    calc_max_drawdown,
    calc_profit_factor,
    get_trade_details,
)
import pandas as pd
from backtest_utils import (
    BacktestOnUniverse,
    Strategy,
    TradingRules,
    create_trade_entries_v2,
    fetch_transform_datasets_on_universe,
)
from constants import AppConstants
from datetime import datetime
from dataset import read_columns_to_mem
from db import get_df_candle_size
from log import LogExceptionContext
from long_short_backtest import get_benchmark_initial_state
from mass_rule_based_backtest import (
    get_backtest_data_range_indexes,
    get_universe_longest_dataset_id,
)
from query_backtest import BacktestQuery
from query_backtest_history import BacktestHistoryQuery
from query_backtest_statistics import BacktestStatisticsQuery
from query_dataset import DatasetQuery
from request_types import BodyRuleBasedMultiStrategy


def run_rule_based_multistrat_backtest(
    log_event_queue: multiprocessing.Queue, body: BodyRuleBasedMultiStrategy
):
    with LogExceptionContext():
        strategies = []
        loop_dataset_id = None
        dataset_candles_time_delta = 1_000_000_000
        benchmark_initial_state = {}
        dataset_timeseries_col = None
        dataset_kline_open_times = pd.DataFrame
        idx_range_start = 0
        idx_range_end = 0

        for strategy in body.strategies:
            result = fetch_transform_datasets_on_universe(
                strategy.datasets,
                strategy.data_transformations,
                strategy.candle_interval,
                strategy.fetch_latest_data,
            )

            dataset_table_name_to_timeseries_col_map = result[
                "dataset_table_name_to_timeseries_col_map"
            ]
            table_names = result["dataset_table_names"]

            longest_dataset_id = get_universe_longest_dataset_id(
                strategy.datasets, strategy.candle_interval
            )
            if longest_dataset_id is None:
                raise Exception("Longest dataset_id was none.")

            longest_dataset = DatasetQuery.fetch_dataset_by_id(longest_dataset_id)
            timeseries_col = longest_dataset.timeseries_column
            kline_open_times = read_columns_to_mem(
                AppConstants.DB_DATASETS,
                longest_dataset.dataset_name,
                [timeseries_col],
            )

            candles_time_delta = get_df_candle_size(
                kline_open_times, timeseries_col, formatted=False
            )

            if candles_time_delta < dataset_candles_time_delta:
                loop_dataset_id = longest_dataset_id
                dataset_candles_time_delta = candles_time_delta
                benchmark_initial_state = get_benchmark_initial_state(
                    table_names, dataset_table_name_to_timeseries_col_map
                )
                (
                    dataset_idx_range_start,
                    dataset_idx_range_end,
                ) = get_backtest_data_range_indexes(
                    kline_open_times, strategy.start_date, strategy.end_date
                )
                idx_range_start = dataset_idx_range_start
                idx_range_end = dataset_idx_range_end
                dataset_kline_open_times = kline_open_times
                dataset_timeseries_col = timeseries_col

            trading_rules = TradingRules(
                trading_fees=strategy.trading_fees_perc,
                use_stop_loss_based_close=strategy.use_stop_loss_based_close,
                use_profit_based_close=strategy.use_profit_based_close,
                use_time_based_close=strategy.use_time_based_close,
                take_profit_threshold_perc=strategy.take_profit_threshold_perc,
                stop_loss_threshold_perc=strategy.stop_loss_threshold_perc,
                short_fee_hourly_perc=strategy.short_fee_hourly,
                max_klines_until_close=strategy.klines_until_close,
                candles_time_delta=candles_time_delta,
            )
            strategies.append(
                Strategy(
                    enter_cond=strategy.open_trade_cond,
                    exit_cond=strategy.close_trade_cond,
                    is_short_selling_strategy=strategy.is_short_selling_strategy,
                    universe_datasets=table_names,
                    leverage_allowed=False,
                    allocation_per_symbol=strategy.allocation_per_symbol / 100,
                    trading_rules=trading_rules,
                    dataset_table_name_to_timeseries_col_map=dataset_table_name_to_timeseries_col_map,
                    longest_dataset=longest_dataset,
                )
            )

        START_BALANCE = 10000

        backtest = BacktestOnUniverse(
            starting_balance=START_BALANCE,
            benchmark_initial_state=benchmark_initial_state,
            candle_time_delta_ms=dataset_candles_time_delta,
            strategies=strategies,
        )

        idx = 0
        processed_bars = 0
        max_bars = idx_range_end - idx_range_start
        sim_start_time = datetime.now()

        for _, row in dataset_kline_open_times.iterrows():
            if idx <= idx_range_start or idx > idx_range_end:
                idx += 1
                continue

            if processed_bars % 50 == 0 and processed_bars > 0:
                datetime_now = datetime.now()
                progress_left = abs((processed_bars / max_bars - 1))
                diff_in_ms = get_diff_in_ms(sim_start_time, datetime_now)
                total_time = max_bars / processed_bars * diff_in_ms

                time_remaining_eta = format_ms_to_human_readable(
                    progress_left * total_time
                )

                log_msg = f"Progress {round((processed_bars / max_bars) * 100, 2)}%. Current profit: {round(backtest.nav - START_BALANCE, 0)}. ETA on completion: {time_remaining_eta}"
                log_event_queue.put(log_msg)

            backtest.tick(kline_open_time=row[dataset_timeseries_col])

            idx += 1
            processed_bars += 1

        backtest_dict = {"name": body.name, "is_multistrategy_backtest": True}

        backtest_id = BacktestQuery.create_entry(backtest_dict)

        profit_factor = calc_profit_factor(backtest.get_all_trades())
        strategy_max_drawdown = calc_max_drawdown(
            [item["portfolio_worth"] for item in backtest.balance_history]
        )
        end_balance = backtest.nav

        strategy_cagr = get_cagr(
            end_balance, START_BALANCE, ms_to_years(backtest.cumulative_time_ms)
        )

        trade_count = len(backtest.get_all_trades())

        trade_details_dict = get_trade_details(backtest.get_all_trades())

        backtest_statistics_dict = {
            "backtest_id": backtest_id,
            "profit_factor": profit_factor,
            "start_balance": START_BALANCE,
            "end_balance": end_balance,
            "result_perc": (end_balance / START_BALANCE - 1) * 100,
            "best_trade_result_perc": trade_details_dict["best_trade_result_perc"],
            "worst_trade_result_perc": trade_details_dict["worst_trade_result_perc"],
            "mean_hold_time_sec": trade_details_dict["mean_hold_time_sec"],
            "mean_return_perc": trade_details_dict["mean_return_perc"],
            "share_of_winning_trades_perc": trade_details_dict[
                "share_of_winning_trades_perc"
            ],
            "share_of_losing_trades_perc": trade_details_dict[
                "share_of_losing_trades_perc"
            ],
            "max_drawdown_perc": strategy_max_drawdown,
            "cagr": strategy_cagr,
            "trade_count": trade_count,
        }

        for item in backtest.balance_history:
            item["kline_open_time"] = int(item["kline_open_time"] / 1000)

        BacktestStatisticsQuery.create_entry(backtest_statistics_dict)
        BacktestHistoryQuery.create_many(backtest_id, backtest.balance_history)

        create_trade_entries_v2(backtest_id, backtest.get_all_trades())

        datetime_now = datetime.now()
        time_elapsed = get_time_elapsed(sim_start_time, datetime_now)

        log_msg = f"Simulation finished. Time elapsed: {time_elapsed}."
        log_event_queue.put(log_msg)
