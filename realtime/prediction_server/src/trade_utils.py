import json
from api.v1.request_types import (
    BodyCreateTrade,
    BodyUpdateTradeClose,
    EnterLongShortPairBody,
)
from constants import TradeDirection
from schema.longshortpair import (
    LongShortPair,
    LongShortPairQuery,
)
from schema.cloudlog import slack_log_close_trade_notif, slack_log_enter_trade_notif
from schema.trade import Trade, TradeQuery
from schema.strategy import Strategy, StrategyQuery
from log import LogExceptionContext, get_logger

from math_utils import (
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

        return trade_update_dict


def close_short_trade(strat: Strategy, trade: Trade, req_body: BodyUpdateTradeClose):
    with LogExceptionContext():
        logger = get_logger()
        update_dict = get_trade_close_dict(strat, trade, req_body)
        slack_log_close_trade_notif(strat, update_dict)
        stringified_dict = json.dumps(update_dict)

        logger.info(
            f"Closing short trade on strat {strat.id} with payload: {stringified_dict}"
        )
        TradeQuery.update_trade(trade.id, update_dict)


def close_long_trade(strat: Strategy, trade: Trade, req_body: BodyUpdateTradeClose):
    with LogExceptionContext():
        logger = get_logger()
        update_dict = get_trade_close_dict(strat, trade, req_body)
        slack_log_close_trade_notif(strat, update_dict)
        stringified_dict = json.dumps(update_dict)
        logger.info(
            f"Closing short trade on strat {strat.id} with payload: {stringified_dict}"
        )
        TradeQuery.update_trade(trade.id, update_dict)


def enter_longshort_trade(pair: LongShortPair, req_body: EnterLongShortPairBody):
    with LogExceptionContext():
        sell_side_trade_body = {
            "open_time_ms": req_body.sell_open_time_ms,
            "quantity": req_body.debt_open_qty_in_base,
            "open_price": req_body.sell_open_price,
            "symbol": req_body.sell_symbol,
            "direction": TradeDirection.SHORT,
        }
        sell_side_trade_id = TradeQuery.create_entry(sell_side_trade_body)

        buy_side_trade_body = {
            "open_time_ms": req_body.buy_open_time_ms,
            "quantity": req_body.buy_open_qty_in_base,
            "open_price": req_body.buy_open_price,
            "symbol": req_body.buy_symbol,
            "direction": TradeDirection.LONG,
        }
        buy_side_trade_id = TradeQuery.create_entry(buy_side_trade_body)

        slack_log_enter_trade_notif(BodyCreateTrade(**sell_side_trade_body))
        slack_log_enter_trade_notif(BodyCreateTrade(**buy_side_trade_body))

        LongShortPairQuery.update_entry(
            pair.id,
            {
                "buy_open_price": req_body.buy_open_price,
                "sell_open_price": req_body.sell_open_price,
                "buy_open_qty_in_base": req_body.buy_open_qty_in_base,
                "buy_open_qty_in_quote": req_body.buy_open_qty_in_base
                * req_body.buy_open_price,
                "sell_open_qty_in_quote": req_body.sell_open_qty_in_quote,
                "debt_open_qty_in_base": req_body.debt_open_qty_in_base,
                "buy_side_trade_id": buy_side_trade_id,
                "sell_side_trade_id": sell_side_trade_id,
            },
        )
