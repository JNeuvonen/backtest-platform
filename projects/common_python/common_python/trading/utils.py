from typing import List


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
