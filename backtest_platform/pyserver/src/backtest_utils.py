from datetime import datetime, timedelta
import math
from typing import List
from math_utils import safe_divide
from query_backtest import BacktestQuery
from query_backtest_history import BacktestHistoryQuery
from constants import ONE_HOUR_IN_MS, CandleSize


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
            overall_gross_wins, overall_gross_losses, None
        ),
        "long_profit_factor": safe_divide(longs_gross_wins, longs_gross_losses, None),
        "short_profit_factor": safe_divide(
            shorts_gross_wins, shorts_gross_losses, None
        ),
    }
