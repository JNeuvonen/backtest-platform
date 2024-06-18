import multiprocessing
from common_python.date import format_ms_to_human_readable, get_diff_in_ms
import pandas as pd
from backtest_utils import (
    BacktestOnUniverse,
    Strategy,
    TradingRules,
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

        print("hello world")
