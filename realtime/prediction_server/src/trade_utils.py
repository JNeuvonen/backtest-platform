from api.v1.request_types import BodyUpdateTradeClose
from schema.trade import Trade
from schema.strategy import Strategy, StrategyQuery
from src.log import LogExceptionContext
from src.math_utils import (
    calc_long_trade_net_result,
    calc_long_trade_perc_result,
    calc_short_trade_net_result,
    calc_short_trade_perc_result,
)


def update_strategy_state(strat: Strategy, req_body: BodyUpdateTradeClose):
    with LogExceptionContext():
        remaining = strat.remaining_position_on_trade - req_body.quantity

        trade_can_be_accounted_as_closed = remaining * req_body.price < 20

        if trade_can_be_accounted_as_closed is True:
            strategy_update_dict = {
                "remaining_position_on_trade": remaining,
                "is_in_position": False,
                "klines_left_till_autoclose": strat.maximum_klines_hold_time,
            }
            StrategyQuery.update_strategy(strat.id, strategy_update_dict)

        if trade_can_be_accounted_as_closed is False:
            strategy_update_dict = {
                "remaining_position_on_trade": remaining,
                "is_in_position": True,
            }
            StrategyQuery.update_strategy(strat.id, strategy_update_dict)


def get_trade_close_dict(strat: Strategy, trade: Trade, req_body: BodyUpdateTradeClose):
    with LogExceptionContext():
        trade_update_dict = {
            "close_price": req_body.price,
            "close_time_ms": req_body.close_time_ms,
        }

        if strat.is_short_selling_strategy is True:
            trade_update_dict["net_result"] = calc_short_trade_net_result(
                req_body.quantity, trade.open_price, req_body.price
            )
            trade_update_dict["percent_result"] = calc_short_trade_perc_result(
                req_body.quantity, trade.open_price, req_body.price
            )
        else:
            trade_update_dict["net_result"] = calc_long_trade_net_result(
                req_body.quantity, trade.open_price, req_body.price
            )
            trade_update_dict["percent_result"] = calc_long_trade_perc_result(
                req_body.quantity, trade.open_price, req_body.price
            )


def close_short_trade(strat: Strategy, trade: Trade, req_body: BodyUpdateTradeClose):
    with LogExceptionContext():
        update_dict = get_trade_close_dict(strat, trade, req_body)
        StrategyQuery.update_strategy(strat.id, update_dict)


def close_long_trade(strat: Strategy, trade: Trade, req_body: BodyUpdateTradeClose):
    with LogExceptionContext():
        update_dict = get_trade_close_dict(strat, trade, req_body)
        StrategyQuery.update_strategy(strat.id, update_dict)
