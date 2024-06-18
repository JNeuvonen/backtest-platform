from datetime import datetime
import json
from typing import List
from api_binance import non_async_save_historical_klines
from backtest_utils import (
    BacktestOnUniverse,
    Strategy,
    TradingRules,
    create_trade_entries_v2,
    fetch_transform_datasets_on_universe,
)
from constants import BINANCE_BACKTEST_PRICE_COL, AppConstants, DomEventChannels
from dataset import (
    get_row_count,
    read_columns_to_mem,
)
from db import exec_python, get_df_candle_size
from log import LogExceptionContext, get_logger
from long_short_backtest import get_benchmark_initial_state
from query_backtest import BacktestQuery
from query_backtest_history import BacktestHistoryQuery
from query_backtest_statistics import BacktestStatisticsQuery
from query_data_transformation import DataTransformationQuery
from query_dataset import DatasetQuery
from request_types import BodyRuleBasedOnUniverse
from utils import PythonCode, get_binance_dataset_tablename
from common_python.trading.utils import (
    calc_profit_factor,
    calc_max_drawdown,
    get_trade_details,
)
from common_python.math import ms_to_years, get_cagr
from datetime import datetime
from common_python.date import (
    get_time_elapsed,
    get_diff_in_ms,
    format_ms_to_human_readable,
)


def get_universe_longest_dataset_id(datasets, candle_interval):
    longest_id = None
    longest_row_count = -1000
    for item in datasets:
        table_name = get_binance_dataset_tablename(item, candle_interval)
        dataset = DatasetQuery.fetch_dataset_by_name(table_name)
        row_count = get_row_count(dataset.dataset_name)

        if longest_row_count < row_count:
            longest_id = dataset.id
            longest_row_count = row_count
    return longest_id


def get_backtest_data_range_indexes(
    kline_open_times, range_start: datetime | None, range_end: datetime | None
):
    if range_end is None and range_start is None:
        return 0, len(kline_open_times)

    if range_end is None and range_start is not None:
        range_start = range_start.timestamp() * 1000
        range_start_idx = 0
        for index, row in kline_open_times.iterrows():
            if row["kline_open_time"] > range_start:
                range_start_idx = index
                break
        return range_start_idx, len(kline_open_times)

    if range_start is None and range_end is not None:
        range_end = range_end.timestamp() * 1000
        range_end_idx = len(kline_open_times) - 1
        for index, row in kline_open_times.iterrows():
            if row["kline_open_time"] > range_end:
                range_end_idx = index - 1
                break
        return 0, range_end_idx + 1

    if range_start is not None and range_end is not None:
        range_start = range_start.timestamp() * 1000
        range_end = range_end.timestamp() * 1000
        range_start_idx = 0
        range_end_idx = len(kline_open_times) - 1
        for index, row in kline_open_times.iterrows():
            if row["kline_open_time"] > range_start:
                range_start_idx = index
                break
        for index, row in kline_open_times.iterrows():
            if row["kline_open_time"] > range_end:
                range_end_idx = index - 1
                break
        return range_start_idx, range_end_idx + 1


def run_rule_based_backtest_on_universe(log_event_queue, body: BodyRuleBasedOnUniverse):
    with LogExceptionContext():
        result = fetch_transform_datasets_on_universe(
            body.datasets,
            body.data_transformations,
            body.candle_interval,
            body.fetch_latest_data,
        )

        table_names = result["dataset_table_names"]
        dataset_table_name_to_id_map = result["dataset_table_name_to_id_map"]
        dataset_id_to_table_name_map = result["dataset_id_to_table_name_map"]
        dataset_table_name_to_timeseries_col_map = result[
            "dataset_table_name_to_timeseries_col_map"
        ]

        longest_dataset_id = get_universe_longest_dataset_id(
            body.datasets, body.candle_interval
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

        benchmark_initial_state = get_benchmark_initial_state(
            table_names, dataset_table_name_to_timeseries_col_map
        )

        idx_range_start, idx_range_end = get_backtest_data_range_indexes(
            kline_open_times, body.start_date, body.end_date
        )

        trading_rules = TradingRules(
            trading_fees=body.trading_fees_perc,
            use_stop_loss_based_close=body.use_stop_loss_based_close,
            use_profit_based_close=body.use_profit_based_close,
            use_time_based_close=body.use_time_based_close,
            take_profit_threshold_perc=body.take_profit_threshold_perc,
            stop_loss_threshold_perc=body.stop_loss_threshold_perc,
            short_fee_hourly_perc=body.short_fee_hourly,
            max_klines_until_close=body.klines_until_close,
            candles_time_delta=candles_time_delta,
        )

        strategies = [
            Strategy(
                enter_cond=body.open_trade_cond,
                exit_cond=body.close_trade_cond,
                is_short_selling_strategy=body.is_short_selling_strategy,
                universe_datasets=table_names,
                leverage_allowed=False,
                allocation_per_symbol=body.allocation_per_symbol / 100,
                trading_rules=trading_rules,
                dataset_table_name_to_timeseries_col_map=dataset_table_name_to_timeseries_col_map,
                longest_dataset=longest_dataset,
            )
        ]

        START_BALANCE = 10000

        backtest = BacktestOnUniverse(
            starting_balance=START_BALANCE,
            benchmark_initial_state=benchmark_initial_state,
            candle_time_delta_ms=candles_time_delta,
            strategies=strategies,
        )

        idx = 0
        processed_bars = 0
        max_bars = idx_range_end - idx_range_start
        sim_start_time = datetime.now()

        for _, row in kline_open_times.iterrows():
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

            backtest.tick(kline_open_time=row[timeseries_col])

            idx += 1
            processed_bars += 1

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

        dataset_ids = []

        for key in dataset_id_to_table_name_map.keys():
            dataset_ids.append(key)

        backtest_dict = {
            "name": body.name,
            "candle_interval": body.candle_interval,
            "open_trade_cond": body.open_trade_cond,
            "close_trade_cond": body.close_trade_cond,
            "use_time_based_close": body.use_time_based_close,
            "use_profit_based_close": body.use_profit_based_close,
            "use_stop_loss_based_close": body.use_stop_loss_based_close,
            "klines_until_close": body.klines_until_close,
            "backtest_date_start_iso": body.start_date.isoformat()
            if body.start_date is not None
            else None,
            "backtest_date_end_iso": body.end_date.isoformat()
            if body.end_date is not None
            else None,
            "is_ran_on_asset_universe": True,
            "is_rule_based_mass_backtest": True,
            "is_short_selling_strategy": body.is_short_selling_strategy,
            "allocation_per_symbol": body.allocation_per_symbol,
            "asset_universe_dataset_ids": json.dumps(dataset_ids),
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
            "share_of_winning_trades_perc": trade_details_dict[
                "share_of_winning_trades_perc"
            ],
            "share_of_losing_trades_perc": trade_details_dict[
                "share_of_losing_trades_perc"
            ],
            "max_drawdown_perc": strategy_max_drawdown,
            "cagr": strategy_cagr,
            "trade_count": trade_count,
            "slippage_perc": body.slippage_perc,
            "short_fee_hourly": body.short_fee_hourly,
            "trading_fees_perc": body.trading_fees_perc,
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
