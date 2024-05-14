import logging
import math
from typing import Dict, List, Set

from api_binance import non_async_save_historical_klines, save_historical_klines
from backtest_utils import (
    calc_long_short_profit_factor,
    calc_max_drawdown,
    get_backtest_data_range_indexes,
    get_cagr,
    get_long_short_trade_details,
    turn_short_fee_perc_to_coeff,
    create_long_short_trades,
)
from code_gen_template import (
    BACKTEST_LONG_SHORT_BUYS_AND_SELLS,
    BACKTEST_LONG_SHORT_CLOSE_TEMPLATE,
)
from constants import BINANCE_BACKTEST_PRICE_COL, AppConstants, DomEventChannels
from dataset import (
    get_row_count,
    read_all_cols_matching_kline_open_times,
    read_columns_to_mem,
    read_dataset_first_row_asc,
)
from db import exec_python, get_df_candle_size, ms_to_years
from log import LogExceptionContext, get_logger
from math_utils import safe_divide
from query_backtest import BacktestQuery
from query_backtest_history import BacktestHistoryQuery
from query_backtest_statistics import BacktestStatisticsQuery
from query_data_transformation import DataTransformationQuery
from query_dataset import DatasetQuery
from request_types import BodyCreateLongShortBacktest
from utils import PythonCode, get_binance_dataset_tablename

START_BALANCE = 10000


def get_longest_dataset_id(backtest_info: BodyCreateLongShortBacktest):
    longest_id = None
    longest_row_count = -1000
    for item in backtest_info.datasets:
        table_name = get_binance_dataset_tablename(item, backtest_info.candle_interval)
        dataset = DatasetQuery.fetch_dataset_by_name(table_name)
        row_count = get_row_count(dataset.dataset_name)

        if longest_row_count < row_count:
            longest_id = dataset.id
            longest_row_count = row_count
    return longest_id


def get_symbol_buy_and_sell_decision(df_row, replacements: Dict):
    code = BACKTEST_LONG_SHORT_BUYS_AND_SELLS

    for key, value in replacements.items():
        code = code.replace(key, str(value))

    results_dict = {"df_row": df_row}
    exec(code, globals(), results_dict)

    ret = {
        "is_valid_buy": results_dict["is_valid_buy"],
        "is_valid_sell": results_dict["is_valid_sell"],
    }
    return ret


def get_benchmark_initial_state(
    datasets: List, dataset_table_name_to_timeseries_col_map: Dict
):
    ret = {}
    for item in datasets:
        timseries_column = dataset_table_name_to_timeseries_col_map[item]
        first_row = read_dataset_first_row_asc(item, timseries_column)
        price = first_row.iloc[0][BINANCE_BACKTEST_PRICE_COL]
        ret[item] = price

    return ret


def get_bar_curr_price(df):
    if not df.empty:
        price = df.iloc[0][BINANCE_BACKTEST_PRICE_COL]
        return price
    return None


def get_datasets_kline_state(
    kline_open_time: int,
    backtest_info: BodyCreateLongShortBacktest,
    dataset_table_name_to_timeseries_col_map: Dict,
    table_names: List[str],
):
    sell_candidates: Set = set()
    buy_candidates: Set = set()

    exec_py_replacements = {
        "{BUY_COND_FUNC}": backtest_info.buy_cond,
        "{SELL_COND_FUNC}": backtest_info.sell_cond,
    }

    table_name_to_df_map = {}

    with LogExceptionContext():
        for item in table_names:
            timeseries_col = dataset_table_name_to_timeseries_col_map[str(item)]
            df = read_all_cols_matching_kline_open_times(
                item, timeseries_col, [kline_open_time]
            )
            table_name_to_df_map[item] = df

            if df.empty is True:
                continue

            df_row = df.iloc[0]
            buy_and_sell_decisions = get_symbol_buy_and_sell_decision(
                df_row, exec_py_replacements
            )

            is_valid_buy = buy_and_sell_decisions["is_valid_buy"]
            is_valid_sell = buy_and_sell_decisions["is_valid_sell"]

            if is_valid_buy is True:
                buy_candidates.add(item)

            if is_valid_sell is True:
                sell_candidates.add(item)

    return {
        "sell_candidates": sell_candidates,
        "buy_candidates": buy_candidates,
        "table_name_to_df_map": table_name_to_df_map,
    }


def run_long_short_backtest(backtest_info: BodyCreateLongShortBacktest):
    dataset_table_names = []
    with LogExceptionContext():
        dataset_table_name_to_id_map = {}
        dataset_id_to_table_name_map = {}
        dataset_table_name_to_timeseries_col_map = {}
        for item in backtest_info.datasets:
            table_name = get_binance_dataset_tablename(
                item, backtest_info.candle_interval
            )
            dataset = DatasetQuery.fetch_dataset_by_name(table_name)

            if dataset is None or backtest_info.fetch_latest_data:
                non_async_save_historical_klines(
                    item, backtest_info.candle_interval, True
                )
                dataset = DatasetQuery.fetch_dataset_by_name(table_name)

            dataset_name = dataset.dataset_name
            dataset_table_name_to_id_map[dataset_name] = dataset.id
            dataset_table_name_to_timeseries_col_map[
                dataset_name
            ] = dataset.timeseries_column
            dataset_id_to_table_name_map[str(dataset.id)] = dataset.dataset_name
            DatasetQuery.update_price_column(dataset_name, BINANCE_BACKTEST_PRICE_COL)
            dataset_table_names.append(dataset_name)

            for data_transform_id in backtest_info.data_transformations:
                transformation = DataTransformationQuery.get_transformation_by_id(
                    data_transform_id
                )
                python_program = PythonCode.on_dataset(
                    dataset_name, transformation.transformation_code
                )
                exec_python(python_program)

        longest_dataset_id = get_longest_dataset_id(backtest_info)
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
            dataset_table_names, dataset_table_name_to_timeseries_col_map
        )

        long_short_backtest = LongShortOnUniverseBacktest(
            backtest_info,
            candles_time_delta,
            dataset_table_name_to_timeseries_col_map,
            benchmark_initial_state,
        )

        if kline_open_times is None:
            raise Exception("Kline_open_times df was none.")

        idx_data_range_start, idx_data_range_end = get_backtest_data_range_indexes(
            kline_open_times, backtest_info
        )
        idx = 0

        logger = get_logger()

        for _, row in kline_open_times.iterrows():
            idx += 1
            if idx <= idx_data_range_start or idx > idx_data_range_end:
                continue

            kline_state = get_datasets_kline_state(
                row[timeseries_col],
                backtest_info,
                dataset_table_name_to_timeseries_col_map,
                dataset_table_names,
            )

            kline_open_time = row[timeseries_col]
            long_short_backtest.process_bar(
                kline_open_time=kline_open_time, kline_state=kline_state
            )

            print(idx)

        profit_factor_dict = calc_long_short_profit_factor(
            long_short_backtest.completed_trades
        )
        max_drawdown_dict = calc_max_drawdown(
            long_short_backtest.positions.position_history
        )
        end_balance = long_short_backtest.positions.net_value

        num_bars = len(long_short_backtest.positions.position_history)
        benchmark_end_balance = long_short_backtest.positions.position_history[
            num_bars - 1
        ]["benchmark_price"]

        strategy_cagr = get_cagr(
            end_balance,
            START_BALANCE,
            ms_to_years(long_short_backtest.stats.cumulative_time),
        )
        benchmark_cagr = get_cagr(
            benchmark_end_balance,
            START_BALANCE,
            ms_to_years(long_short_backtest.stats.cumulative_time),
        )
        strat_weighted_time_exp = (
            long_short_backtest.stats.position_held_time
            / long_short_backtest.stats.cumulative_time
        )
        strategy_risk_adjusted_cagr = safe_divide(
            strategy_cagr, strat_weighted_time_exp, 0
        )
        trade_count = len(long_short_backtest.completed_trades)
        trade_details_dict = get_long_short_trade_details(
            long_short_backtest.completed_trades
        )

        backtest_dict = {
            "name": backtest_info.name,
            "candle_interval": get_df_candle_size(
                kline_open_times, timeseries_col, formatted=True
            ),
            "long_short_buy_cond": backtest_info.buy_cond,
            "long_short_sell_cond": backtest_info.sell_cond,
            "long_short_exit_cond": backtest_info.exit_cond,
            "is_long_short_strategy": True,
            "use_time_based_close": backtest_info.use_time_based_close,
            "use_profit_based_close": backtest_info.use_profit_based_close,
            "use_stop_loss_based_close": backtest_info.use_stop_loss_based_close,
            "klines_until_close": backtest_info.klines_until_close,
            "backtest_range_start": backtest_info.backtest_data_range[0],
            "backtest_range_end": backtest_info.backtest_data_range[1],
        }

        backtest_id = BacktestQuery.create_entry(backtest_dict)

        backtest_statistics_dict = {
            "backtest_id": backtest_id,
            "asset_universe_size": len(dataset_table_names),
            "profit_factor": profit_factor_dict["strategy_profit_factor"],
            "long_side_profit_factor": profit_factor_dict["long_profit_factor"],
            "short_side_profit_factor": profit_factor_dict["short_profit_factor"],
            "gross_profit": profit_factor_dict["total_gross_wins"],
            "gross_loss": profit_factor_dict["total_gross_losses"],
            "start_balance": START_BALANCE,
            "end_balance": end_balance,
            "result_perc": (end_balance / START_BALANCE - 1) * 100,
            "take_profit_threshold_perc": backtest_info.take_profit_threshold_perc,
            "stop_loss_threshold_perc": backtest_info.stop_loss_threshold_perc,
            "best_trade_result_perc": trade_details_dict["best_trade_result_perc"],
            "worst_trade_result_perc": trade_details_dict["worst_trade_result_perc"],
            "mean_hold_time_sec": trade_details_dict["mean_hold_time_sec"],
            "mean_return_perc": trade_details_dict["mean_return_perc"],
            "buy_and_hold_result_net": benchmark_end_balance - START_BALANCE,
            "buy_and_hold_result_perc": (benchmark_end_balance / START_BALANCE - 1)
            * 100,
            "share_of_winning_trades_perc": trade_details_dict[
                "share_of_winning_trades_perc"
            ],
            "share_of_losing_trades_perc": trade_details_dict[
                "share_of_losing_trades_perc"
            ],
            "max_drawdown_perc": max_drawdown_dict["drawdown_strategy_perc"],
            "benchmark_drawdown_perc": max_drawdown_dict["drawdown_benchmark_perc"],
            "cagr": strategy_cagr,
            "market_exposure_time": strat_weighted_time_exp,
            "risk_adjusted_return": strategy_risk_adjusted_cagr,
            "buy_and_hold_cagr": benchmark_cagr,
            "trade_count": trade_count,
            "slippage_perc": backtest_info.slippage_perc,
            "short_fee_hourly": backtest_info.short_fee_hourly,
            "trading_fees_perc": backtest_info.trading_fees_perc,
            "best_long_side_trade_result_perc": trade_details_dict[
                "best_long_trade_perc"
            ],
            "worst_long_side_trade_result_perc": trade_details_dict[
                "worst_long_trade_perc"
            ],
            "best_short_side_trade_result_perc": trade_details_dict[
                "best_short_trade_perc"
            ],
            "worst_short_side_trade_result_perc": trade_details_dict[
                "worst_short_trade_perc"
            ],
        }

        BacktestStatisticsQuery.create_entry(backtest_statistics_dict)

        for item in long_short_backtest.positions.position_history:
            item["kline_open_time"] = int(item["kline_open_time"] / 1000)

        BacktestHistoryQuery.create_many(
            backtest_id, long_short_backtest.positions.position_history
        )
        create_long_short_trades(
            backtest_id,
            long_short_backtest.completed_trades,
        )

        logger = get_logger()
        logger.log(
            f"Finished mass pair-trade backtest. Profit: {(end_balance/START_BALANCE - 1) * 100}%",
            logging.INFO,
            True,
            True,
            DomEventChannels.REFETCH_COMPONENT.value,
        )


class BacktestRules:
    def __init__(
        self, backtest_details: BodyCreateLongShortBacktest, candles_time_delta
    ):
        self.buy_cond = backtest_details.buy_cond
        self.sell_cond = backtest_details.sell_cond
        self.exit_cond = backtest_details.exit_cond

        self.use_profit_based_close = backtest_details.use_profit_based_close
        self.use_stop_loss_based_close = backtest_details.use_stop_loss_based_close
        self.use_time_based_close = backtest_details.use_time_based_close

        self.max_klines_until_close = backtest_details.klines_until_close
        self.take_profit_threshold_perc = backtest_details.take_profit_threshold_perc
        self.stop_loss_threshold_perc = backtest_details.stop_loss_threshold_perc

        self.trading_fees = (backtest_details.trading_fees_perc) / 100
        self.short_fee_coeff = turn_short_fee_perc_to_coeff(
            backtest_details.short_fee_hourly, candles_time_delta
        )

        self.max_simultaneous_positions = backtest_details.max_simultaneous_positions
        self.max_leverage_ratio = backtest_details.max_leverage_ratio


class BacktestStats:
    def __init__(self, candles_time_delta: int):
        self.cumulative_time = 0
        self.position_held_time = 0
        self.candles_time_delta = candles_time_delta


class PositionManager:
    def __init__(self, usdt_start_balance: float):
        self.usdt_balance = usdt_start_balance
        self.net_value = usdt_start_balance
        self.usdt_debt = 0

        self.open_positions_total_debt = 0
        self.open_positions_total_long = 0
        self.debt_to_net_value_ratio = 0.0

        self.position_history: List = []

    def close_long(self, proceedings: float):
        self.usdt_balance += proceedings

    def close_short(self, proceedings: float):
        self.usdt_balance -= proceedings

    def update(
        self, positions_debt, positions_long, kline_open_time, benchmark_eq_value
    ):
        self.net_value = self.usdt_balance + positions_long - positions_debt
        self.open_positions_total_long = positions_long
        self.open_positions_total_debt = positions_debt
        self.debt_to_net_value_ratio = positions_debt / self.net_value

        tick = {
            "total_debt": positions_debt,
            "total_long": positions_long,
            "kline_open_time": kline_open_time,
            "debt_to_net_value_ratio": self.debt_to_net_value_ratio,
            "portfolio_worth": self.net_value,
            "benchmark_price": benchmark_eq_value,
        }
        self.position_history.append(tick)


class BenchmarkManager:
    def __init__(self, start_balance, benchmark_initial_state: Dict):
        num_pairs = len(benchmark_initial_state)
        allocation_per_pair = start_balance / num_pairs
        positions = {}
        for key, value in benchmark_initial_state.items():
            positions[key] = allocation_per_pair / value

        self.positions = positions
        self.equity_value = start_balance
        self.prices = benchmark_initial_state

    def update(self, kline_state_dict):
        curr_equity = 0.0
        for key, value in kline_state_dict["table_name_to_df_map"].items():
            if value.empty:
                continue
            price = get_bar_curr_price(value)
            self.prices[key] = price

        for key, value in self.positions.items():
            pos_eq_value = value * self.prices[key]
            curr_equity += pos_eq_value

        self.equity_value = curr_equity


class PairTrade:
    def __init__(
        self,
        buy_id: int,
        buy_amount_base: float,
        buy_amount_quote: float,
        sell_id: int,
        sell_proceedings_quote: float,
        debt_amount_base: float,
        trade_open_time,
        sell_price: float,
        buy_price: float,
        on_trade_open_acc_net_value: float,
    ):
        self.buy_id = buy_id
        self.sell_id = sell_id
        self.buy_amount_base = buy_amount_base
        self.buy_amount_quote = buy_amount_quote
        self.sell_proceedings_quote = sell_proceedings_quote
        self.debt_amount_base = debt_amount_base
        self.trade_open_time = trade_open_time
        self.on_open_sell_price = sell_price
        self.on_open_buy_price = buy_price
        self.on_trade_open_acc_net_value = on_trade_open_acc_net_value
        self.balance_history: List[Dict] = []
        self.candles_held = 0
        self.prev_buy_side_price = 0
        self.prev_sell_side_price = 0
        self.prev_buy_side_df = None
        self.prev_sell_side_df = None

    def get_buy_side_price(self, price):
        if price is not None:
            self.prev_buy_side_price = price
        if price is None:
            return self.prev_buy_side_price
        return price

    def get_sell_side_price(self, price):
        if price is not None:
            self.prev_sell_side_price = price
        if price is None:
            return self.prev_sell_side_price
        return price

    def get_buy_side_price_from_df(self, df):
        if df.empty:
            return self.prev_buy_side_price
        return df.iloc[0][BINANCE_BACKTEST_PRICE_COL]

    def get_sell_side_price_from_df(self, df):
        if df.empty:
            return self.prev_sell_side_price
        return df.iloc[0][BINANCE_BACKTEST_PRICE_COL]

    def get_results_dict(self, kline_state: Dict):
        buy_df = kline_state["table_name_to_df_map"][self.buy_id]
        sell_df = kline_state["table_name_to_df_map"][self.sell_id]

        if buy_df.empty:
            buy_df = self.prev_buy_side_df
        else:
            self.prev_buy_side_df = buy_df

        if sell_df.empty:
            sell_df = self.prev_sell_side_df
        else:
            self.prev_sell_side_df = sell_df

        return {"buy_df": buy_df.iloc[0], "sell_df": sell_df.iloc[0]}


class CompletedPairTrade:
    def __init__(
        self,
        buy_id: int,
        sell_id: int,
        long_open_price: float,
        long_close_price: float,
        short_open_price: float,
        short_close_price: float,
        on_open_long_side_amount_quote: float,
        on_close_long_side_amount_quote: float,
        on_open_short_side_amount_quote: float,
        on_close_short_side_amount_quote: float,
        on_trade_open_acc_net_value: float,
        balance_history: List[float],
        open_time: int,
        close_time: int,
    ):
        self.buy_id = buy_id
        self.sell_id = sell_id

        self.long_open_price = long_open_price
        self.long_close_price = long_close_price

        self.short_open_price = short_open_price
        self.short_close_price = short_close_price

        self.balance_history = balance_history

        self.long_side_gross_result = (
            on_close_long_side_amount_quote - on_open_long_side_amount_quote
        )
        self.long_side_perc_result = (
            safe_divide(
                on_close_long_side_amount_quote,
                on_open_long_side_amount_quote,
                fallback=0,
            )
            - 1
        ) * 100
        self.short_side_gross_result = (
            on_open_short_side_amount_quote - on_close_short_side_amount_quote
        )
        self.short_side_perc_result = (
            safe_divide(
                (on_open_short_side_amount_quote - on_close_short_side_amount_quote),
                on_open_short_side_amount_quote,
                fallback=0,
            )
            - 1
        ) * 100
        self.trade_gross_result = (
            self.long_side_gross_result + self.short_side_gross_result
        )

        self.perc_result = (
            safe_divide(self.trade_gross_result, on_trade_open_acc_net_value, 0) * 100
        )
        self.open_time = open_time
        self.close_time = close_time


class LongShortOnUniverseBacktest:
    def __init__(
        self,
        backtest_details: BodyCreateLongShortBacktest,
        candles_time_delta: int,
        dataset_table_name_to_timeseries_col_map: Dict,
        benchmark_initial_state: Dict,
    ):
        self.rules = BacktestRules(backtest_details, candles_time_delta)
        self.stats = BacktestStats(candles_time_delta)
        self.positions = PositionManager(START_BALANCE)
        self.benchmark = BenchmarkManager(START_BALANCE, benchmark_initial_state)
        self.dataset_table_name_to_timeseries_col_map = (
            dataset_table_name_to_timeseries_col_map
        )

        self.active_pairs: List[PairTrade] = []
        self.completed_trades: List = []

    def form_trading_pairs(self, buy_and_sell_candidates: Dict):
        valid_buys = list(buy_and_sell_candidates["buy_candidates"])
        valid_sells = list(buy_and_sell_candidates["sell_candidates"])
        n_valid_sells = len(valid_sells)

        ret = []

        for i in range(len(valid_buys)):
            if i < n_valid_sells:
                valid_buy = valid_buys[i]
                valid_sell = valid_sells[i]
                ret.append({"sell": valid_sell, "buy": valid_buy})
            else:
                break

        return ret

    def get_available_new_pos_size(self):
        current_debt_ratio = self.positions.debt_to_net_value_ratio
        max_total_allocation = self.rules.max_leverage_ratio
        max_invidual_allocation = (
            max_total_allocation / self.rules.max_simultaneous_positions
        )

        if max_total_allocation - current_debt_ratio < 0.0:
            return 0

        new_allocation_size = min(
            max_invidual_allocation, max_total_allocation - current_debt_ratio
        )
        return new_allocation_size * self.positions.net_value

    def remove_pairs_already_in_trade(self, available_pairs):
        active_trade_ids = {trade.buy_id for trade in self.active_pairs} | {
            trade.sell_id for trade in self.active_pairs
        }

        filtered_pairs = [
            item
            for item in available_pairs
            if item["buy"] not in active_trade_ids
            and item["sell"] not in active_trade_ids
        ]

        return filtered_pairs

    def should_exit_by_condition(self, kline_state: Dict, pair: PairTrade):
        code = BACKTEST_LONG_SHORT_CLOSE_TEMPLATE
        replacements = {"{EXIT_PAIR_TRADE_FUNC}": self.rules.exit_cond}

        for key, value in replacements.items():
            code = code.replace(key, str(value))

        results_dict = pair.get_results_dict(kline_state)
        exec(code, globals(), results_dict)

        should_close_trade = results_dict["should_close_trade"]
        return should_close_trade

    def get_close_long_proceedings(self, buy_df, pair: PairTrade):
        close_long_amount = (
            pair.get_buy_side_price_from_df(buy_df)
            * pair.buy_amount_base
            * self.trading_fees_coeff_reduce_amount()
        )
        return close_long_amount

    def trading_fees_coeff_reduce_amount(self):
        return 1 - self.rules.trading_fees

    def trading_fees_coeff_increase_amount(self):
        return self.rules.trading_fees + 1

    def get_close_short_amount(self, sell_df, pair: PairTrade):
        close_short_amount = (
            pair.get_sell_side_price_from_df(sell_df)
            * pair.debt_amount_base
            * self.trading_fees_coeff_increase_amount()
        )
        return close_short_amount

    def get_trade_enter_long_amount(self, usdt_size: float, price: float):
        amount = usdt_size / price
        return amount * self.trading_fees_coeff_reduce_amount()

    def close_pair_trade(
        self, kline_state: Dict, pair: PairTrade, kline_open_time: int
    ):
        buy_df = kline_state["table_name_to_df_map"][pair.buy_id]
        sell_df = kline_state["table_name_to_df_map"][pair.sell_id]

        long_close_price = pair.get_buy_side_price(get_bar_curr_price(buy_df))
        short_close_price = pair.get_sell_side_price(get_bar_curr_price(sell_df))

        sell_long_proceedings = self.get_close_long_proceedings(buy_df, pair)
        required_for_close_short = self.get_close_short_amount(sell_df, pair)

        completed_trade = CompletedPairTrade(
            buy_id=pair.buy_id,
            sell_id=pair.sell_id,
            long_open_price=pair.on_open_buy_price,
            long_close_price=long_close_price,
            short_open_price=pair.on_open_sell_price,
            short_close_price=short_close_price,
            balance_history=pair.balance_history,
            on_open_long_side_amount_quote=pair.buy_amount_quote,
            on_open_short_side_amount_quote=pair.sell_proceedings_quote,
            on_close_long_side_amount_quote=sell_long_proceedings,
            on_close_short_side_amount_quote=required_for_close_short,
            on_trade_open_acc_net_value=pair.on_trade_open_acc_net_value,
            close_time=kline_open_time,
            open_time=pair.trade_open_time,
        )

        self.positions.close_long(sell_long_proceedings)
        self.positions.close_short(required_for_close_short)
        self.completed_trades.append(completed_trade)

    def is_pos_held_for_max_time(self, pair_trade):
        if (
            self.rules.use_time_based_close
            and self.rules.max_klines_until_close <= pair_trade.candles_held
        ):
            return True
        return False

    def is_pair_going_to_reenter(self, pair_trade, kline_state):
        buy_id = pair_trade.buy_id
        sell_id = pair_trade.sell_id

        return (
            buy_id in kline_state["buy_candidates"]
            and sell_id in kline_state["sell_candidates"]
        )

    def should_trading_stop_loss_close(self, pair_trade):
        if self.rules.use_stop_loss_based_close is False:
            return False

        if len(pair_trade.balance_history) == 0:
            return False

        profit_ath = pair_trade.balance_history[0]["net_profit_in_quote"]
        size = pair_trade.sell_proceedings_quote
        peak_result_perc = (profit_ath / size) * 100

        for balance_history in pair_trade.balance_history:
            curr_profit_perc = (balance_history["net_profit_in_quote"] / size) * 100

            if (
                curr_profit_perc < -1 * self.rules.stop_loss_threshold_perc
                or peak_result_perc - self.rules.stop_loss_threshold_perc
                > curr_profit_perc
            ):
                return True

            profit_ath = max(curr_profit_perc, peak_result_perc)

        return False

    def check_for_pair_trade_close(self, kline_state, kline_open_time):
        active_pairs = []
        for item in self.active_pairs:
            should_exit_by_cond = self.should_exit_by_condition(kline_state, item)
            should_exit_by_time_based = self.is_pos_held_for_max_time(item)
            should_trading_stop_loss = self.should_trading_stop_loss_close(item)

            is_pair_going_to_reenter = self.is_pair_going_to_reenter(item, kline_state)

            if should_exit_by_cond is True:
                self.close_pair_trade(kline_state, item, kline_open_time)
            elif (
                should_exit_by_time_based is True and is_pair_going_to_reenter is False
            ):
                self.close_pair_trade(kline_state, item, kline_open_time)
            elif should_trading_stop_loss is True and is_pair_going_to_reenter is False:
                self.close_pair_trade(kline_state, item, kline_open_time)
            else:
                active_pairs.append(item)
        self.active_pairs = active_pairs

    def enter_pair_trade(self, kline_open_time, kline_state, pair):
        buy_id = pair["buy"]
        sell_id = pair["sell"]

        buy_df = kline_state["table_name_to_df_map"][buy_id]
        sell_df = kline_state["table_name_to_df_map"][sell_id]

        buy_price = get_bar_curr_price(buy_df)
        sell_price = get_bar_curr_price(sell_df)

        usdt_size = self.get_available_new_pos_size()

        debt_amount_base = usdt_size / sell_price
        sell_proceedings_quote = (
            debt_amount_base * sell_price * self.trading_fees_coeff_reduce_amount()
        )
        buy_amount_base = self.get_trade_enter_long_amount(
            sell_proceedings_quote, buy_price
        )

        new_pair_trade = PairTrade(
            buy_id=buy_id,
            sell_id=sell_id,
            debt_amount_base=debt_amount_base,
            buy_amount_base=buy_amount_base,
            buy_amount_quote=buy_amount_base * buy_price,
            sell_proceedings_quote=sell_proceedings_quote,
            trade_open_time=kline_open_time,
            sell_price=sell_price,
            buy_price=buy_price,
            on_trade_open_acc_net_value=self.positions.net_value,
        )
        self.active_pairs.append(new_pair_trade)

    def enter_trades(self, kline_open_time, kline_state, enter_trade_pairs):
        for pair in enter_trade_pairs:
            self.enter_pair_trade(kline_open_time, kline_state, pair)

    def update_pos_held_time(self):
        capital_usage_coeff = safe_divide(
            len(self.active_pairs), self.rules.max_simultaneous_positions, 0
        )
        bar_pos_held_time = math.floor(
            capital_usage_coeff * self.stats.candles_time_delta
        )

        self.stats.position_held_time += bar_pos_held_time
        self.stats.cumulative_time += self.stats.candles_time_delta

    def update_active_pairs(self, kline_open_time: int, kline_state: Dict):
        for item in self.active_pairs:
            item.candles_held += 1

            buy_df = kline_state["table_name_to_df_map"][item.buy_id]
            sell_df = kline_state["table_name_to_df_map"][item.sell_id]

            buy_side_price = item.get_buy_side_price(get_bar_curr_price(buy_df))
            sell_side_price = item.get_sell_side_price(get_bar_curr_price(sell_df))

            total_long = buy_side_price * item.buy_amount_base
            total_debt = sell_side_price * item.debt_amount_base

            pos_net_value = total_long - total_debt

            item.balance_history.append(
                {
                    "kline_open_time": kline_open_time / 1000,
                    "net_profit_in_quote": pos_net_value,
                    "long_value_in_quote": total_long,
                    "short_debt_in_quote": total_debt,
                }
            )

    def update_acc_state(self, kline_open_time: int, kline_state: Dict):
        total_debt = 0.0
        total_longs = 0.0

        for item in self.active_pairs:
            item.debt_amount_base *= self.rules.short_fee_coeff

        for item in self.active_pairs:
            buy_df = kline_state["table_name_to_df_map"][item.buy_id]
            sell_df = kline_state["table_name_to_df_map"][item.sell_id]

            buy_side_price = item.get_buy_side_price(get_bar_curr_price(buy_df))
            sell_side_price = item.get_sell_side_price(get_bar_curr_price(sell_df))

            total_longs += buy_side_price * item.buy_amount_base
            total_debt += sell_side_price * item.debt_amount_base

        self.benchmark.update(kline_state)
        self.positions.update(
            positions_debt=total_debt,
            positions_long=total_longs,
            kline_open_time=kline_open_time,
            benchmark_eq_value=self.benchmark.equity_value,
        )
        self.update_active_pairs(kline_open_time, kline_state)
        self.update_pos_held_time()

    def process_bar(self, kline_open_time: int, kline_state: Dict):
        self.update_acc_state(kline_open_time, kline_state)
        self.check_for_pair_trade_close(kline_state, kline_open_time)

        pairs_available = self.remove_pairs_already_in_trade(
            self.form_trading_pairs(kline_state)
        )
        self.enter_trades(kline_open_time, kline_state, pairs_available)
