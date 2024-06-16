from datetime import datetime, timedelta
import math
from common_python.trading.utils import (
    is_stop_loss_hit,
    is_take_profit_hit,
    calc_trade_net_result,
    calc_trade_perc_result,
    Trade,
)
from typing import Dict, List, Optional, Set
from code_gen_template import BACKTEST_MANUAL_TEMPLATE, LOAD_MODEL_TEMPLATE
from dataset import read_all_cols_matching_kline_open_times
from db import candlesize_to_timedelta, exec_python
from math_utils import safe_divide
from query_backtest import BacktestQuery
from query_backtest_history import BacktestHistoryQuery
from constants import BINANCE_BACKTEST_PRICE_COL, ONE_HOUR_IN_MS, CandleSize
from query_data_transformation import DataTransformation
from query_dataset import Dataset
from query_pair_trade import PairTradeQuery
from query_trade import TradeQuery
from request_types import BodyMLBasedBacktest
from utils import PythonCode


START_BALANCE = 10000


def get_bar_curr_price(df):
    if not df.empty:
        price = df.iloc[0][BINANCE_BACKTEST_PRICE_COL]
        return price
    return None


def get_backtest_profit_factor_comp(trades):
    gross_profit = 0.0
    gross_loss = 0.0

    for item in trades:
        if item["net_result"] >= 0.0:
            gross_profit += item["net_result"]
        else:
            gross_loss += abs(item["net_result"])

    return (
        gross_profit / gross_loss if gross_loss != 0 else None,
        gross_profit,
        gross_loss,
    )


def get_backtest_trade_details(trades):
    BEST_TRADE_RESULT_START = -100
    WORST_TRADE_RESULT_START = 100

    best_trade_result_perc = BEST_TRADE_RESULT_START
    worst_trade_result_perc = WORST_TRADE_RESULT_START
    total_winning_trades = 0
    total_losing_trades = 0

    for item in trades:
        if item["net_result"] > 0.0:
            total_winning_trades += 1

            if best_trade_result_perc < item["percent_result"]:
                best_trade_result_perc = item["percent_result"]

        if item["net_result"] < 0.0:
            total_losing_trades += 1

            if worst_trade_result_perc > item["percent_result"]:
                worst_trade_result_perc = item["percent_result"]

    total_trades = len(trades) if len(trades) > 0 else 1

    return (
        (total_winning_trades / total_trades) * 100,
        (total_losing_trades / total_trades) * 100,
        best_trade_result_perc
        if best_trade_result_perc != BEST_TRADE_RESULT_START
        else None,
        worst_trade_result_perc
        if worst_trade_result_perc != WORST_TRADE_RESULT_START
        else None,
    )


def find_max_drawdown(balances):
    START_MAX_DRAWDOWN = 1.0
    max_drawdown = START_MAX_DRAWDOWN
    last_high_balance = None

    for item in balances:
        if last_high_balance is None:
            last_high_balance = item["portfolio_worth"]
            continue

        if last_high_balance < item["portfolio_worth"]:
            last_high_balance = item["portfolio_worth"]
            continue

        curr_drawdown = item["portfolio_worth"] / last_high_balance

        if curr_drawdown < max_drawdown:
            max_drawdown = curr_drawdown

    return (max_drawdown - 1) * 100 if max_drawdown != START_MAX_DRAWDOWN else None


def get_cagr(end_balance, start_balance, years):
    return (end_balance / start_balance) ** (1 / years) - 1


def turn_short_fee_perc_to_coeff(short_fee_hourly_perc: float, candles_time_delta):
    hours_in_candle = 0

    if candles_time_delta == ONE_HOUR_IN_MS:
        return 1 + (short_fee_hourly_perc / 100)
    elif candles_time_delta > ONE_HOUR_IN_MS:
        hours_in_candle = candles_time_delta / ONE_HOUR_IN_MS
        return ((short_fee_hourly_perc / 100) + 1) ** hours_in_candle
    elif candles_time_delta < ONE_HOUR_IN_MS:
        exponent = ONE_HOUR_IN_MS / candles_time_delta
        return (1 + (short_fee_hourly_perc / 100)) ** (1 / exponent)
    return 0


def calc_long_side_trade_res_perc(open_price, close_price):
    if open_price == 0:
        return 0

    return (close_price / open_price - 1) * 100


def calc_short_side_trade_res_perc(open_price, close_price):
    if open_price == 0:
        return 0
    return (close_price / open_price - 1) * -1 * 100


def get_dataset_exit_and_enter_decisions(df_row, replacements: Dict):
    code = BACKTEST_MANUAL_TEMPLATE

    for key, value in replacements.items():
        code = code.replace(key, str(value))

    results_dict = {"df_row": df_row}
    exec(code, globals(), results_dict)
    ret = {
        "should_open": results_dict["should_open_trade"],
        "should_exit": results_dict["should_close_trade"],
    }
    return ret


def get_kline_decisions_v2(
    kline_open_time: int,
    open_cond: str,
    exit_cond: str,
    dataset_table_name_to_timeseries_col_map: Dict,
    table_names: List[str],
):
    enters: Set = set()
    exits: Set = set()

    exec_py_replacements = {
        "{OPEN_TRADE_FUNC}": open_cond,
        "{CLOSE_TRADE_FUNC}": exit_cond,
    }
    table_name_to_df_map = {}

    for item in table_names:
        timeseries_col = dataset_table_name_to_timeseries_col_map[str(item)]
        df = read_all_cols_matching_kline_open_times(
            item, timeseries_col, [kline_open_time]
        )
        table_name_to_df_map[item] = df

        if df.empty is True:
            continue

        df_row = df.iloc[0]
        kline_trading_decisions = get_dataset_exit_and_enter_decisions(
            df_row, exec_py_replacements
        )

        should_open = kline_trading_decisions["should_open"]
        should_exit = kline_trading_decisions["should_exit"]

        if should_open is True:
            enters.add(item)

        if should_exit is True:
            exits.add(item)

    return {
        "enters": list(enters),
        "exits": list(exits),
        "table_name_to_df_map": table_name_to_df_map,
    }


def get_trade_details(trades):
    cumulative_hold_time = 0
    cumulative_returns = 0.0
    num_winning_trades = 0
    num_losing_trades = 0

    worst_trade_result_perc = float("inf")
    best_trade_result_perc = float("-inf")

    for item in trades:
        if item.trade_gross_result > 0:
            num_winning_trades += 1
        if item.trade_gross_result < 0:
            num_losing_trades += 1

        cumulative_hold_time += item.close_time - item.open_time
        cumulative_returns += item.perc_result

        worst_trade_result_perc = min(item.perc_result, worst_trade_result_perc)
        best_trade_result_perc = max(item.perc_result, best_trade_result_perc)

    num_total_trades = num_winning_trades + num_losing_trades

    best_trade_result_perc = (
        None if best_trade_result_perc == float("-inf") else best_trade_result_perc
    )
    worst_trade_result_perc = (
        None if worst_trade_result_perc == float("inf") else worst_trade_result_perc
    )

    return {
        "best_trade_result_perc": best_trade_result_perc,
        "worst_trade_result_perc": worst_trade_result_perc,
        "share_of_winning_trades_perc": safe_divide(
            num_winning_trades, num_total_trades, 0
        )
        * 100,
        "share_of_losing_trades_perc": safe_divide(
            num_losing_trades, num_total_trades, 0
        )
        * 100,
        "mean_return_perc": safe_divide(cumulative_returns, num_total_trades, 0),
        "mean_hold_time_sec": safe_divide(cumulative_hold_time, num_total_trades, 0)
        / 1000,
    }


def get_long_short_trade_details(trades):
    cumulative_hold_time = 0
    cumulative_returns = 0.0
    num_winning_trades = 0
    num_losing_trades = 0

    worst_trade_result_perc = float("inf")
    best_trade_result_perc = float("-inf")

    best_long_trade_perc = float("-inf")
    worst_long_trade_perc = float("inf")

    best_short_trade_perc = float("-inf")
    worst_short_trade_perc = float("inf")

    for item in trades:
        if item.trade_gross_result > 0:
            num_winning_trades += 1
        if item.trade_gross_result < 0:
            num_losing_trades += 1

        long_side_res_perc = calc_long_side_trade_res_perc(
            item.long_open_price, item.long_close_price
        )
        short_side_res_perc = calc_short_side_trade_res_perc(
            item.short_open_price, item.short_close_price
        )

        cumulative_returns += item.perc_result
        cumulative_hold_time += item.close_time - item.open_time

        best_trade_result_perc = max(best_trade_result_perc, item.perc_result)
        worst_trade_result_perc = min(worst_trade_result_perc, item.perc_result)

        best_long_trade_perc = max(best_long_trade_perc, long_side_res_perc)
        worst_long_trade_perc = min(worst_long_trade_perc, long_side_res_perc)

        best_short_trade_perc = max(best_short_trade_perc, short_side_res_perc)
        worst_short_trade_perc = min(worst_short_trade_perc, short_side_res_perc)

    best_trade_result_perc = (
        None if best_trade_result_perc == float("-inf") else best_trade_result_perc
    )
    worst_trade_result_perc = (
        None if worst_trade_result_perc == float("inf") else worst_trade_result_perc
    )
    best_long_trade_perc = (
        None if best_long_trade_perc == float("-inf") else best_long_trade_perc
    )
    worst_long_trade_perc = (
        None if worst_long_trade_perc == float("inf") else worst_long_trade_perc
    )
    best_short_trade_perc = (
        None if best_short_trade_perc == float("-inf") else best_short_trade_perc
    )
    worst_short_trade_perc = (
        None if worst_short_trade_perc == float("inf") else worst_short_trade_perc
    )

    num_total_trades = num_winning_trades + num_losing_trades

    return {
        "best_trade_result_perc": best_trade_result_perc,
        "worst_trade_result_perc": worst_trade_result_perc,
        "best_long_trade_perc": best_long_trade_perc,
        "worst_long_trade_perc": worst_long_trade_perc,
        "best_short_trade_perc": best_short_trade_perc,
        "worst_short_trade_perc": worst_short_trade_perc,
        "share_of_winning_trades_perc": safe_divide(
            num_winning_trades, num_total_trades, 0
        )
        * 100,
        "share_of_losing_trades_perc": safe_divide(
            num_losing_trades, num_total_trades, 0
        )
        * 100,
        "mean_return_perc": safe_divide(cumulative_returns, num_total_trades, 0),
        "mean_hold_time_sec": safe_divide(cumulative_hold_time, num_total_trades, 0)
        / 1000,
    }


def get_mass_sim_backtests_equity_curves(list_of_ids: List[int], candle_interval: str):
    ret = []
    first_kline_open_time_ms = math.inf
    last_kline_open_time_ms = -math.inf

    for item in list_of_ids:
        first_and_last_kline = (
            BacktestHistoryQuery.get_first_last_kline_times_by_backtest_id(item)
        )

        first_kline = first_and_last_kline["first_kline_open_time"]
        last_kline = first_and_last_kline["last_kline_open_time"]

        if first_kline is None or last_kline is None:
            continue

        first_kline_open_time_ms = min(first_kline_open_time_ms, first_kline)
        last_kline_open_time_ms = max(last_kline_open_time_ms, last_kline)

    start_date = datetime.utcfromtimestamp(first_kline_open_time_ms / 1000)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.utcfromtimestamp(last_kline_open_time_ms / 1000)
    current_date = start_date

    kline_open_times = []

    while current_date <= end_date:
        kline_open_time_ms = int(current_date.timestamp() * 1000)
        kline_open_times.append(kline_open_time_ms)
        current_date += timedelta(days=1)

    for item in list_of_ids:
        if candle_interval == CandleSize.ONE_DAY:
            balance_history = (
                BacktestHistoryQuery.get_entries_by_backtest_id_sorted_partial(item)
            )
        else:
            balance_history = BacktestHistoryQuery.get_ticks_by_kline_open_times(
                item, kline_open_times
            )
        if len(balance_history) == 0:
            continue
        ret.append({str(item): balance_history})

    return ret


def get_backtest_id_to_dataset_name_map(list_of_ids: List[int]):
    ret = {}

    for item in list_of_ids:
        dataset_name = BacktestQuery.fetch_dataset_name_by_id(item)

        ret[str(item)] = dataset_name

    return ret


def calc_long_short_profit_factor(trades):
    longs_gross_wins = 0.0
    longs_gross_losses = 0.0
    shorts_gross_wins = 0.0
    shorts_gross_losses = 0.0

    overall_gross_wins = 0.0
    overall_gross_losses = 0.0

    for trade in trades:
        if trade.long_side_gross_result >= 0.0:
            longs_gross_wins += trade.long_side_gross_result
        else:
            longs_gross_losses += trade.long_side_gross_result

        if trade.short_side_gross_result >= 0.0:
            shorts_gross_wins += trade.short_side_gross_result
        else:
            shorts_gross_losses += trade.short_side_gross_result

        if trade.trade_gross_result >= 0.0:
            overall_gross_wins += trade.trade_gross_result
        else:
            overall_gross_losses += trade.trade_gross_result

    return {
        "strategy_profit_factor": safe_divide(
            overall_gross_wins, abs(overall_gross_losses), None
        ),
        "long_profit_factor": safe_divide(
            longs_gross_wins, abs(longs_gross_losses), None
        ),
        "short_profit_factor": safe_divide(
            shorts_gross_wins, abs(shorts_gross_losses), None
        ),
        "total_gross_losses": abs(overall_gross_losses),
        "total_gross_wins": abs(overall_gross_wins),
    }


def calc_max_drawdown(balances):
    max_drawdown_strategy = 1000
    max_drawdown_benchmark = 1000

    strategy_peak_balance = balances[0]["portfolio_worth"]
    benchmark_peak_balance = balances[0]["benchmark_price"]

    for item in balances:
        strat_net_value = item["portfolio_worth"]
        benchmark_net_value = item["benchmark_price"]

        strategy_peak_balance = max(strat_net_value, strategy_peak_balance)
        benchmark_peak_balance = max(benchmark_net_value, benchmark_peak_balance)

        max_drawdown_strategy = min(
            strat_net_value / strategy_peak_balance, max_drawdown_strategy
        )
        max_drawdown_benchmark = min(
            benchmark_net_value / benchmark_peak_balance, max_drawdown_benchmark
        )

    return {
        "drawdown_benchmark_perc": abs(max_drawdown_benchmark - 1) * 100,
        "drawdown_strategy_perc": abs(max_drawdown_strategy - 1) * 100,
    }


def calculate_index_from_percentage(data_length, percentage):
    return math.floor(data_length * (percentage / 100))


def get_backtest_data_range_indexes(dataset_df, backtest_info):
    start_percentage, end_percentage = backtest_info.backtest_data_range[:2]

    assert start_percentage is not None, "Backtest data range start is missing"
    assert end_percentage is not None, "Backtest data range end is missing"

    backtest_data_range_start = calculate_index_from_percentage(
        len(dataset_df), start_percentage
    )
    backtest_data_range_end = calculate_index_from_percentage(
        len(dataset_df), end_percentage
    )

    return backtest_data_range_start, backtest_data_range_end


def get_long_trade_from_completed_pair_trade(pair_trade, backtest_id):
    return {
        "is_short_trade": False,
        "open_price": pair_trade.long_open_price,
        "close_price": pair_trade.long_close_price,
        "open_time": int(pair_trade.open_time / 1000),
        "close_time": int(pair_trade.close_time / 1000),
        "net_result": pair_trade.long_side_gross_result,
        "percent_result": pair_trade.long_side_perc_result,
        "dataset_name": pair_trade.buy_id,
        "backtest_id": backtest_id,
    }


def get_short_trade_from_completed_pair_trade(pair_trade, backtest_id):
    return {
        "is_short_trade": True,
        "open_price": pair_trade.short_open_price,
        "close_price": pair_trade.short_close_price,
        "open_time": int(pair_trade.open_time / 1000),
        "close_time": int(pair_trade.close_time / 1000),
        "net_result": pair_trade.short_side_gross_result,
        "percent_result": pair_trade.short_side_perc_result,
        "dataset_name": pair_trade.buy_id,
        "backtest_id": backtest_id,
    }


def get_pair_trade_entry(pair_trade, backtest_id, long_trade_id, short_trade_id):
    return {
        "backtest_id": backtest_id,
        "buy_trade_id": long_trade_id,
        "sell_trade_id": short_trade_id,
        "gross_result": pair_trade.trade_gross_result,
        "percent_result": pair_trade.perc_result,
        "open_time": int(pair_trade.open_time / 1000),
        "close_time": int(pair_trade.close_time / 1000),
    }


def trade_entry_body(backtest_id, completed_trade):
    return {
        "backtest_id": backtest_id,
        "open_price": completed_trade.open_price,
        "close_price": completed_trade.close_price,
        "open_time": completed_trade.open_time / 1000,
        "close_time": completed_trade.close_time / 1000,
        "net_result": completed_trade.trade_gross_result,
        "percent_result": completed_trade.perc_result,
        "is_short_trade": False if completed_trade.was_long_trade is True else True,
    }


def create_trade_entries(backtest_id, completed_trades):
    for item in completed_trades:
        TradeQuery.create_entry(trade_entry_body(backtest_id, item))


def create_long_short_trades(backtest_id, completed_trades):
    for item in completed_trades:
        long_trade = get_long_trade_from_completed_pair_trade(item, backtest_id)
        long_trade_id = TradeQuery.create_entry(long_trade)

        short_trade = get_short_trade_from_completed_pair_trade(item, backtest_id)
        short_trade_id = TradeQuery.create_entry(short_trade)

        pair_trade = get_pair_trade_entry(
            item, backtest_id, long_trade_id, short_trade_id
        )

        PairTradeQuery.create_entry(pair_trade)


def run_transformations_on_dataset(
    transformations: List[DataTransformation], dataset: Dataset
):
    for transformation in transformations:
        python_program = PythonCode.on_dataset(
            dataset.dataset_name, transformation.transformation_code
        )
        exec_python(python_program)


def exec_load_model(model_class_code, train_job_id, epoch_nr, x_input_size):
    template = LOAD_MODEL_TEMPLATE
    replacements = {
        "{MODEL_CLASS}": model_class_code,
        "{X_INPUT_SIZE}": x_input_size,
        "{TRAIN_JOB_ID}": train_job_id,
        "{EPOCH_NR}": epoch_nr,
    }

    for key, value in replacements.items():
        template = template.replace(key, str(value))

    local_variables = {}
    exec(template, local_variables, local_variables)
    model = local_variables.get("model")
    if model is None:
        raise Exception("Unable to load model")
    return model


class MLBasedBacktestRules:
    def __init__(self, backtest_details: BodyMLBasedBacktest, candle_interval):
        self.use_profit_based_close = backtest_details.use_profit_based_close
        self.use_stop_loss_based_close = backtest_details.use_stop_loss_based_close
        self.use_time_based_close = backtest_details.use_time_based_close

        self.max_klines_until_close = backtest_details.klines_until_close
        self.take_profit_threshold_perc = backtest_details.take_profit_threshold_perc
        self.stop_loss_threshold_perc = backtest_details.stop_loss_threshold_perc

        self.trading_fees = (backtest_details.trading_fees_perc) / 100
        self.trading_fees_coeff = 1 - (backtest_details.trading_fees_perc / 100)

        self.short_fee_coeff = turn_short_fee_perc_to_coeff(
            backtest_details.short_fee_hourly,
            candlesize_to_timedelta(candle_interval) * 1000,
        )


class MLBasedBacktestStats:
    def __init__(self, candle_interval):
        self.cumulative_time = 0
        self.position_held_time = 0
        self.candles_time_delta = candlesize_to_timedelta(candle_interval) * 1000


class MLBasedPositionManager:
    def __init__(self, usdt_start_balance: float):
        self.usdt_balance = usdt_start_balance
        self.net_value = usdt_start_balance
        self.usdt_debt = 0

        self.open_positions_total_debt = 0
        self.open_positions_total_long = 0
        self.debt_to_net_value_ratio = 0.0
        self.position_history: List = []

    def open_long(self, proceedings: float):
        self.usdt_balance -= proceedings

    def open_short(self, proceedings: float):
        self.usdt_balance += proceedings

    def close_long(self, proceedings: float):
        self.usdt_balance += proceedings

    def close_short(self, proceedings: float):
        self.usdt_balance -= proceedings
        self.open_positions_total_debt = 0.0

    def update(
        self, positions_debt, positions_long, kline_open_time, benchmark_eq_value
    ):
        self.net_value = self.usdt_balance + positions_long - positions_debt
        self.open_positions_total_long = positions_long
        self.open_positions_total_debt = positions_debt
        self.debt_to_net_value_ratio = safe_divide(
            positions_debt, self.net_value, fallback=0
        )

        tick = {
            "total_debt": positions_debt,
            "total_long": positions_long,
            "kline_open_time": kline_open_time,
            "debt_to_net_value_ratio": self.debt_to_net_value_ratio,
            "portfolio_worth": self.net_value,
            "benchmark_price": benchmark_eq_value,
        }
        self.position_history.append(tick)


class MLBasedBenchmarkManager:
    def __init__(self, start_balance, first_price):
        self.position = start_balance / first_price


class MLBasedActiveTrade:
    def __init__(
        self,
        enter_price: float,
        size_in_base: float,
        is_side_long: bool,
        debt_amount_base: float,
        short_proceedings: float,
        kline_open_time: float,
        size_in_quote: float,
    ):
        self.enter_price = enter_price
        self.size_in_base = size_in_base
        self.is_side_long = is_side_long
        self.candles_held = 0
        self.balance_history: List[Dict] = []
        self.debt_amount_base = debt_amount_base
        self.short_proceedings = short_proceedings
        self.trade_open_time = kline_open_time

        self.size_in_quote = size_in_quote

    def update(self, kline_open_time, price):
        self.candles_held += 1
        self.balance_history.append(
            {
                "kline_open_time": kline_open_time,
                "price": price,
                "debt_amount_base": self.debt_amount_base,
                "size_in_base": self.size_in_base,
            }
        )


class MLBasedCompletedTrade:
    def __init__(
        self,
        open_price: float,
        close_price: float,
        was_long_trade: bool,
        open_time,
        close_time,
        balance_history: List,
        on_open_amount_quote: float,
        on_close_amount_quote: float,
    ):
        self.balance_history = balance_history
        self.open_time = open_time
        self.close_time = close_time
        self.was_long_trade = was_long_trade
        self.open_price = open_price
        self.close_price = close_price

        if was_long_trade is True:
            self.trade_gross_result = on_close_amount_quote - on_open_amount_quote
            self.perc_result = (
                safe_divide(on_close_amount_quote, on_open_amount_quote, fallback=0) - 1
            ) * 100

        else:
            self.trade_gross_result = on_open_amount_quote - on_close_amount_quote
            self.perc_result = (
                safe_divide(
                    (on_open_amount_quote - on_close_amount_quote),
                    on_open_amount_quote,
                    fallback=0,
                )
                * 100
            )


class MLBasedBacktest:
    def __init__(
        self, backtest_details: BodyMLBasedBacktest, dataset, first_price: float
    ):
        self.rules = MLBasedBacktestRules(backtest_details, dataset.interval)
        self.stats = MLBasedBacktestStats(dataset.interval)
        self.position = MLBasedPositionManager(START_BALANCE)
        self.benchmark = MLBasedBenchmarkManager(START_BALANCE, first_price)
        self.active_trade: Optional[MLBasedActiveTrade] = None
        self.completed_trades: List[MLBasedCompletedTrade] = []

    def update_pos_held_time(self):
        if self.active_trade is not None:
            self.stats.position_held_time += self.stats.candles_time_delta
        self.stats.cumulative_time += self.stats.candles_time_delta

    def update_acc_state(self, kline_open_time: int, price: float):
        benchmark_eq_value = self.benchmark.position * price

        if self.active_trade is not None and self.active_trade.is_side_long is False:
            self.active_trade.debt_amount_base *= self.rules.short_fee_coeff

        if self.active_trade is not None:
            debt_amount = self.active_trade.debt_amount_base * price
            long_amount = self.active_trade.size_in_base * price
            self.position.update(
                debt_amount, long_amount, kline_open_time, benchmark_eq_value
            )
            self.active_trade.update(kline_open_time, price)

        else:
            self.position.update(0, 0, kline_open_time, benchmark_eq_value)
        self.update_pos_held_time()

    def trading_fees_coeff_reduce_amount(self):
        return 1 - self.rules.trading_fees

    def trading_fees_coeff_increase_amount(self):
        return self.rules.trading_fees + 1

    def close_trade(self, kline_open_time: int, price: float):
        if self.active_trade is None:
            return

        if self.active_trade.is_side_long is True:
            close_size_in_quote = (
                self.active_trade.size_in_base
                * price
                * self.trading_fees_coeff_reduce_amount()
            )
        else:
            close_size_in_quote = (
                self.active_trade.debt_amount_base
                * price
                * self.trading_fees_coeff_increase_amount()
            )

        completed_trade = MLBasedCompletedTrade(
            self.active_trade.enter_price,
            price,
            self.active_trade.is_side_long,
            self.active_trade.trade_open_time,
            kline_open_time,
            self.active_trade.balance_history,
            self.active_trade.size_in_quote,
            close_size_in_quote,
        )
        self.completed_trades.append(completed_trade)

        if self.active_trade.is_side_long is True:
            self.position.close_long(close_size_in_quote)
        else:
            self.position.close_short(close_size_in_quote)

        self.active_trade = None

    def check_close_by_trading_stop_loss(self):
        if self.active_trade is None:
            return False

        if self.rules.use_stop_loss_based_close is False:
            return False

        price_ath = self.active_trade.enter_price
        price_atl = self.active_trade.enter_price

        for item in self.active_trade.balance_history:
            if self.active_trade.is_side_long:
                if item["price"] < price_ath and (
                    abs((item["price"] / price_ath - 1) * 100)
                    >= self.rules.stop_loss_threshold_perc
                ):
                    return True
                price_ath = max(price_ath, item["price"])
            else:
                if (
                    item["price"] > price_atl
                    and (item["price"] / price_atl - 1) * 100
                    >= self.rules.stop_loss_threshold_perc
                ):
                    return True
                price_atl = min(price_atl, item["price"])
        return False

    def check_close_by_take_profit(self, price):
        if self.active_trade is None:
            return False
        if self.rules.use_profit_based_close is False:
            return False

        enter_price = self.active_trade.enter_price

        if self.active_trade.is_side_long is True:
            return (
                price / enter_price - 1
            ) * 100 >= self.rules.take_profit_threshold_perc
        else:
            return (
                1 - price / enter_price
            ) * 100 >= self.rules.take_profit_threshold_perc

    def check_close_by_time_based(self):
        if self.active_trade is None:
            return False
        if self.rules.use_time_based_close is False:
            return False

        return self.active_trade.candles_held >= self.rules.max_klines_until_close

    def check_for_trade_close(
        self, kline_open_time: int, price: float, trading_decisions: Dict
    ):
        if self.active_trade is None:
            return

        should_close_long_trade = trading_decisions["should_close_long_trade"]
        should_close_short_trade = trading_decisions["should_close_short_trade"]

        if self.active_trade.is_side_long is True and should_close_long_trade is True:
            self.close_trade(kline_open_time, price)
            return

        if self.active_trade.is_side_long is False and should_close_short_trade is True:
            self.close_trade(kline_open_time, price)
            return

        if self.check_close_by_time_based() is True:
            self.close_trade(kline_open_time, price)
            return

        if self.check_close_by_trading_stop_loss() is True:
            self.close_trade(kline_open_time, price)
            return

        if self.check_close_by_take_profit(price) is True:
            self.close_trade(kline_open_time, price)
            return

    def enter_long_trade(self, kline_open_time: int, price: float):
        size_in_base = self.position.usdt_balance / price
        self.position.open_long(self.position.usdt_balance)
        self.active_trade = MLBasedActiveTrade(
            price,
            size_in_base * self.trading_fees_coeff_reduce_amount(),
            True,
            0.0,
            0.0,
            kline_open_time,
            size_in_base * price,
        )

    def enter_short_trade(self, kline_open_time: int, price: float):
        size_in_base = self.position.usdt_balance / price
        short_proceedings = size_in_base * self.rules.trading_fees_coeff * price
        self.position.open_short(short_proceedings)
        self.active_trade = MLBasedActiveTrade(
            price,
            0.0,
            False,
            size_in_base,
            short_proceedings,
            kline_open_time,
            size_in_base * price * self.trading_fees_coeff_reduce_amount(),
        )

    def check_for_enter_trade(
        self, kline_open_time: int, price: float, trading_decisions: Dict
    ):
        if self.active_trade is not None:
            return

        should_enter_long_trade = trading_decisions["should_open_long_trade"]
        should_enter_short_trade = trading_decisions["should_open_short_trade"]

        if should_enter_short_trade is True:
            self.enter_short_trade(kline_open_time, price)

        if should_enter_long_trade is True:
            self.enter_long_trade(kline_open_time, price)

    def process_bar(self, kline_open_time: int, price: float, trading_decisions: Dict):
        self.update_acc_state(kline_open_time, price)
        self.check_for_trade_close(kline_open_time, price, trading_decisions)
        self.check_for_enter_trade(kline_open_time, price, trading_decisions)


def calc_profit_factor(trades):
    gross_profit = 0.0
    gross_losses = 0.0

    for item in trades:
        if item.trade_gross_result >= 0:
            gross_profit += item.trade_gross_result
        else:
            gross_losses += abs(item.trade_gross_result)

    return safe_divide(gross_profit, gross_losses, fallback=None)


TABLE_NAME_TO_DF_MAP = "table_name_to_df_map"


class TradingRules:
    def __init__(
        self,
        trading_fees: float,
        use_stop_loss_based_close: bool,
        use_profit_based_close: bool,
        use_time_based_close: bool,
        take_profit_threshold_perc: float,
        stop_loss_threshold_perc: float,
        short_fee_hourly_perc: float,
        max_klines_until_close: int | None,
        candles_time_delta,
    ):
        self.trading_fees = trading_fees / 100
        self.use_stop_loss_based_close = use_stop_loss_based_close
        self.use_profit_based_close = use_profit_based_close
        self.use_time_based_close = use_time_based_close
        self.take_profit_threshold_perc = take_profit_threshold_perc
        self.stop_loss_threshold_perc = stop_loss_threshold_perc
        self.short_fee_hourly_perc = short_fee_hourly_perc
        self.max_klines_until_close = max_klines_until_close
        self.short_fee_coeff = turn_short_fee_perc_to_coeff(
            short_fee_hourly_perc, candles_time_delta
        )


class Position:
    def __init__(
        self,
        dataset_name: str,
        quantity: float,
        open_quote_quantity: float,
        open_time_ms: int,
        open_price: float,
        is_debt: bool,
    ):
        self.dataset_name = dataset_name
        self.open_quantity = quantity
        self.open_quote_quantity = open_quote_quantity
        self.quantity = quantity
        self.open_time_ms = open_time_ms
        self.open_price = open_price
        self.is_debt = is_debt
        self.candles_held = 0
        self.prev_price = open_price
        self.balance_history: List[Dict] = [
            {
                "position_size": quantity * open_price,
                "kline_open_time": open_time_ms,
                "price": open_price,
            }
        ]
        self.is_closed = False
        self.close_proceedings = 0

    def get_price_safe(self, price: float | None):
        if price is None:
            return self.prev_price
        return price

    def update(self, kline_open_time: int, new_price: float | None):
        self.candles_held += 1

        if new_price is not None:
            self.prev_price = new_price

        price = self.get_price_safe(new_price)
        position_size = self.quantity * price

        self.balance_history.append(
            {
                "position_size": position_size,
                "kline_open_time": kline_open_time,
                "price": price,
            }
        )


class Strategy:
    def __init__(
        self,
        enter_cond: str,
        exit_cond: str,
        is_short_selling_strategy: bool,
        universe_datasets: List[str],
        leverage_allowed: float,
        trading_rules: TradingRules,
        dataset_table_name_to_timeseries_col_map: Dict,
        allocation_per_symbol: float,
        longest_dataset: Dataset,
    ):
        self.enter_cond = enter_cond
        self.exit_cond = exit_cond
        self.is_short_selling_strategy = is_short_selling_strategy
        self.universe_datasets = universe_datasets
        self.is_leverage_allowed = leverage_allowed
        self.trading_rules = trading_rules
        self.allocation_per_symbol = allocation_per_symbol
        self.dataset_table_name_to_timeseries_col_map = (
            dataset_table_name_to_timeseries_col_map
        )
        self.kline_state: Dict = {}
        self.table_name_to_df_map: Dict = {}
        self.positions: List[Position] = []
        self.longest_dataset = longest_dataset
        self.trades: List[Trade] = []
        self.is_no_data_on_curr_row = False

    def process_bar(self, kline_open_time: int):
        df = read_all_cols_matching_kline_open_times(
            self.longest_dataset.dataset_name,
            self.longest_dataset.timeseries_column,
            [kline_open_time],
        )
        if not df.empty:
            self.kline_state = get_kline_decisions_v2(
                kline_open_time,
                self.enter_cond,
                self.exit_cond,
                self.dataset_table_name_to_timeseries_col_map,
                self.universe_datasets,
            )
        else:
            self.is_no_data_on_curr_row = True

    def trading_fees_coeff_reduce_amount(self):
        return 1 - self.trading_rules.trading_fees

    def trading_fees_coeff_increase_amount(self):
        return self.trading_rules.trading_fees + 1

    def is_dataset_already_in_pos(self, dataset: str):
        for item in self.positions:
            if item.dataset_name == dataset:
                return True
        return False

    def enter_trade(self, dataset: str, kline_open_time: int, usdt_allocation: float):
        price = get_bar_curr_price(self.kline_state[TABLE_NAME_TO_DF_MAP][dataset])

        if price is None:
            return False

        quantity = usdt_allocation / price

        if self.is_short_selling_strategy is False:
            quantity = quantity * self.trading_fees_coeff_reduce_amount()

        position = Position(
            dataset_name=dataset,
            quantity=quantity,
            open_quote_quantity=usdt_allocation,
            open_time_ms=kline_open_time,
            open_price=price,
            is_debt=self.is_short_selling_strategy,
        )
        self.positions.append(position)
        return True

    def tick(self, kline_open_time: int):
        for item in self.positions:
            if self.is_no_data_on_curr_row is False:
                price = get_bar_curr_price(
                    self.kline_state[TABLE_NAME_TO_DF_MAP][item.dataset_name]
                )
                item.update(kline_open_time=kline_open_time, new_price=price)

            if self.is_short_selling_strategy is True:
                item.quantity *= self.trading_rules.short_fee_coeff

    def get_assets(self):
        if self.is_short_selling_strategy is True:
            return 0

        assets = 0.0

        for item in self.positions:
            price = get_bar_curr_price(
                self.kline_state[TABLE_NAME_TO_DF_MAP][item.dataset_name]
            )

            if price is not None:
                assets += price * item.quantity
        return assets

    def get_liabilities(self):
        if self.is_short_selling_strategy is False:
            return 0

        liabilities = 0.0

        for item in self.positions:
            price = get_bar_curr_price(
                self.kline_state[TABLE_NAME_TO_DF_MAP][item.dataset_name]
            )
            if price is not None:
                liabilities += price * item.quantity
        return liabilities

    def should_close_by_exit_cond(self, position: Position):
        exits = self.kline_state["exits"]
        for item in exits:
            if item == position.dataset_name:
                return True
        return False

    def is_pos_held_for_max_time(self, position: Position):
        if self.trading_rules.use_time_based_close is True:
            return position.candles_held >= self.trading_rules.max_klines_until_close
        return False

    def should_close_by_stop_loss(self, position: Position):
        if self.trading_rules.use_stop_loss_based_close is False:
            return False

        if len(position.balance_history) == 0:
            return False

        return is_stop_loss_hit(
            [item["price"] for item in position.balance_history],
            self.trading_rules.stop_loss_threshold_perc,
            self.is_short_selling_strategy,
        )

    def should_close_by_take_profit(self, position: Position):
        if self.trading_rules.use_profit_based_close is False:
            return False

        if len(position.balance_history) == 0:
            return False

        last_price = position.balance_history[len(position.balance_history) - 1][
            "price"
        ]
        open_price = position.balance_history[0]["price"]

        return is_take_profit_hit(
            last_price,
            open_price,
            self.trading_rules.take_profit_threshold_perc,
            self.is_short_selling_strategy,
        )


class BacktestOnUniverse:
    def __init__(
        self,
        trading_rules: TradingRules,
        starting_balance: float,
        benchmark_initial_state: Dict,
        candle_time_delta_ms: int,
        strategies: List[Strategy],
        max_margin_ratio: int = 1,
    ):
        self.trading_rules = trading_rules
        self.starting_balance = starting_balance
        self.candle_time_delta_ms = candle_time_delta_ms
        self.cumulative_time_ms = 0
        self.positions: List[Position] = []
        self.cash_balance = starting_balance
        self.nav = starting_balance
        self.cash_debt = 0
        self.max_margin_ratio = max_margin_ratio
        self.current_margin_ratio = 0
        self.strategies = strategies
        self.balance_history: List[Dict] = []

    def update(self, kline_open_time: int):
        total_assets = 0.0
        total_liabilities = 0.0

        for item in self.strategies:
            item.tick(kline_open_time)

        for item in self.strategies:
            if item.is_short_selling_strategy is True:
                total_liabilities += item.get_liabilities()
            else:
                total_assets += item.get_assets()

        nav = self.cash_balance + total_assets - total_liabilities
        self.nav = nav
        self.current_margin_ratio = safe_divide(total_liabilities, self.nav, 0.0)

        tick = {
            "total_debt": total_liabilities,
            "total_long": total_assets,
            "kline_open_time": kline_open_time,
            "debt_to_net_value_ratio": self.current_margin_ratio,
            "portfolio_worth": self.nav,
        }
        self.cumulative_time_ms += self.candle_time_delta_ms
        self.balance_history.append(tick)

    def close_trade(self, strategy: Strategy, position: Position, kline_open_time: int):
        last_price = position.balance_history[len(position.balance_history) - 1][
            "price"
        ]
        net_result = calc_trade_net_result(
            position.open_price,
            last_price,
            position.open_quantity,
            position.quantity,
            strategy.is_short_selling_strategy,
        )
        perc_result = calc_trade_perc_result(
            position.open_price,
            last_price,
            position.open_quantity,
            position.quantity,
            strategy.is_short_selling_strategy,
        )

        trade = Trade(
            open_price=position.open_price,
            open_time=position.open_time_ms,
            close_price=last_price,
            close_time=kline_open_time,
            net_result=net_result,
            percent_result=perc_result,
            direction="SHORT" if strategy.is_short_selling_strategy else "LONG",
            dataset_name=position.dataset_name,
        )
        strategy.trades.append(trade)
        position.close_proceedings = last_price * position.quantity
        position.is_closed = True

    def close_trades(self, kline_open_time: int):
        for item in self.strategies:
            for position in item.positions:
                should_close = False

                if item.should_close_by_exit_cond(position) is True:
                    should_close = True
                if item.should_close_by_stop_loss(position) is True:
                    should_close = True
                if item.should_close_by_take_profit(position) is True:
                    should_close = True
                if item.is_pos_held_for_max_time(position) is True:
                    should_close = True

                if should_close is True:
                    if item.is_short_selling_strategy is False:
                        self.close_trade(item, position, kline_open_time)
                        close_proceedings = (
                            position.close_proceedings
                            * item.trading_fees_coeff_reduce_amount()
                        )

                        if self.cash_debt > 0:
                            if self.cash_debt >= close_proceedings:
                                self.cash_debt -= close_proceedings
                            else:
                                remaining_after_close_debt = (
                                    close_proceedings - self.cash_debt
                                )
                                self.cash_debt = 0.0
                                self.cash_balance += remaining_after_close_debt
                        else:
                            self.cash_balance += close_proceedings
                    else:
                        self.close_trade(item, position, kline_open_time)
                        close_proceedings = (
                            position.close_proceedings
                            * item.trading_fees_coeff_increase_amount()
                        )

                        if self.cash_balance >= close_proceedings:
                            self.cash_balance -= close_proceedings
                        else:
                            needed_debt = close_proceedings - self.cash_balance
                            self.cash_debt += needed_debt
                            self.cash_balance = 0.0

    def get_short_size_usdt(self, strategy: Strategy):
        allocation_per_symbol = strategy.allocation_per_symbol

        if self.current_margin_ratio >= self.max_margin_ratio:
            return 0

        if allocation_per_symbol + self.current_margin_ratio >= self.max_margin_ratio:
            return (self.max_margin_ratio - self.current_margin_ratio) * self.nav

        return allocation_per_symbol * self.nav

    def get_long_size_usdt(self, strategy: Strategy):
        allocated_size = strategy.allocation_per_symbol * self.nav
        if self.cash_balance >= allocated_size:
            return allocated_size
        return self.cash_balance

    def enter_trades(self, kline_open_time: int):
        for item in self.strategies:
            if item.is_no_data_on_curr_row is True:
                continue

            enters = item.kline_state["enters"]
            for enter_dataset in enters:
                if item.is_dataset_already_in_pos(enter_dataset) is False:
                    if item.is_short_selling_strategy is True:
                        size = self.get_short_size_usdt(item)
                        if size < 30:
                            continue

                        if (
                            item.enter_trade(
                                enter_dataset,
                                kline_open_time,
                                size,
                            )
                            is True
                        ):
                            self.cash_balance += (
                                size * item.trading_fees_coeff_reduce_amount()
                            )
                    else:
                        size = self.get_long_size_usdt(item)
                        if size < 30:
                            continue

                        if (
                            item.enter_trade(
                                enter_dataset,
                                kline_open_time,
                                size,
                            )
                            is True
                        ):
                            self.cash_balance -= size

    def cleanup(self):
        for item in self.strategies:
            positions = []
            for position in item.positions:
                if position.is_closed is False:
                    positions.append(position)
            item.positions = positions

    def tick(self, kline_open_time: int):
        for item in self.strategies:
            item.process_bar(kline_open_time)
        self.update(kline_open_time)
        self.close_trades(kline_open_time)
        self.enter_trades(kline_open_time)
        self.cleanup()

    def get_all_trades(self):
        trades = []
        for item in self.strategies:
            trades += item.trades
        return trades


def create_trade_entries_v2(backtest_id: int, trades: List[Trade]):
    for item in trades:
        trade_dict = item.to_dict()
        trade_dict["backtest_id"] = backtest_id
        trade_dict["is_short_trade"] = True if item.direction == "SHORT" else False
        trade_dict["open_time"] = trade_dict["open_time"] / 1000
        trade_dict["close_time"] = trade_dict["close_time"] / 1000
        TradeQuery.create_entry(trade_dict)
