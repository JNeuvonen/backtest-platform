from datetime import datetime, timedelta
import math
from typing import Dict, List, Optional
from code_gen_template import LOAD_MODEL_TEMPLATE
from db import candlesize_to_timedelta, exec_python
from math_utils import safe_divide
from query_backtest import BacktestQuery
from query_backtest_history import BacktestHistoryQuery
from constants import ONE_HOUR_IN_MS, CandleSize
from query_data_transformation import DataTransformation
from query_dataset import Dataset
from query_pair_trade import PairTradeQuery
from query_trade import TradeQuery
from request_types import BodyMLBasedBacktest
from utils import PythonCode


START_BALANCE = 10000


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
