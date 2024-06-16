from typing import List

from common_python.math import safe_divide


class Trade:
    def __init__(
        self,
        open_price,
        close_price,
        open_time,
        close_time,
        direction,
        net_result,
        percent_result,
        dataset_name,
    ):
        self.open_price = open_price
        self.close_price = close_price
        self.open_time = open_time
        self.close_time = close_time
        self.direction = direction
        self.net_result = net_result
        self.percent_result = percent_result
        self.dataset_name = dataset_name

    def to_dict(self):
        return {
            "open_price": self.open_price,
            "close_price": self.close_price,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "direction": self.direction,
            "net_result": self.net_result,
            "percent_result": self.percent_result,
            "dataset_name": self.dataset_name,
        }


def is_stop_loss_hit(prices: List[int], stop_loss_threshold, is_short_selling_strategy):
    if len(prices) == 0:
        return False
    if is_short_selling_strategy:
        price_ath = prices[0]

        for item in prices:
            if ((price_ath / item) - 1) * 100 > stop_loss_threshold:
                return True
            if item > price_ath:
                price_ath = item
    else:
        price_atl = prices[0]

        for item in prices:
            if ((item / price_atl) - 1) * 100 > stop_loss_threshold:
                return True
            if item < price_atl:
                price_atl = item
    return False


def is_take_profit_hit(
    last_price, open_price, take_profit_threshold, is_short_selling_strategy
):
    if is_short_selling_strategy:
        profit_percentage = ((open_price - last_price) / open_price) * 100
    else:
        profit_percentage = ((last_price - open_price) / open_price) * 100

    return profit_percentage >= take_profit_threshold


def calc_trade_net_result(
    open_price: float,
    close_price: float,
    open_quantity: float,
    close_quantity: float,
    is_short_selling_strategy: bool,
):
    if is_short_selling_strategy is True:
        proceeds_received_on_open = open_quantity * open_price
        money_needed_to_close = close_quantity * close_price
        return proceeds_received_on_open - money_needed_to_close
    else:
        net_result = (close_price - open_price) * open_quantity
        return net_result


def calc_trade_perc_result(
    open_price: float,
    close_price: float,
    open_quantity: float,
    close_quantity: float,
    is_short_selling_strategy: bool,
):
    if is_short_selling_strategy:
        proceeds_received_on_open = open_quantity * open_price
        money_needed_to_close = close_quantity * close_price
        net_result = proceeds_received_on_open - money_needed_to_close
        percentage_result = (net_result / proceeds_received_on_open) * 100
    else:
        initial_investment = open_quantity * open_price
        net_result = (close_price - open_price) * open_quantity
        percentage_result = (net_result / initial_investment) * 100

    return percentage_result


def calc_profit_factor(trades: List[Trade]):
    gross_profit = 0.0
    gross_losses = 0.0

    for item in trades:
        if item.net_result >= 0:
            gross_profit += item.net_result
        else:
            gross_losses += item.net_result

    return safe_divide(gross_profit, abs(gross_losses), None)


def calc_max_drawdown(equity: List[float]):
    max_drawdown_strategy = 1000

    strategy_peak_balance = equity[0]

    for strat_net_value in equity:
        strategy_peak_balance = max(strat_net_value, strategy_peak_balance)
        max_drawdown_strategy = min(
            strat_net_value / strategy_peak_balance, max_drawdown_strategy
        )

    return abs(max_drawdown_strategy - 1) * 100


def get_trade_details(trades: List[Trade]):
    cumulative_hold_time = 0
    cumulative_returns = 0.0
    num_winning_trades = 0
    num_losing_trades = 0

    worst_trade_result_perc = float("inf")
    best_trade_result_perc = float("-inf")

    for item in trades:
        if item.net_result > 0:
            num_winning_trades += 1
        if item.net_result < 0:
            num_losing_trades += 1

        cumulative_hold_time += item.close_time - item.open_time
        cumulative_returns += item.percent_result

        worst_trade_result_perc = min(item.percent_result, worst_trade_result_perc)
        best_trade_result_perc = max(item.percent_result, best_trade_result_perc)

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
