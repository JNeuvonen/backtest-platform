import json
from api.v1.request_types import (
    BodyCreateTrade,
    BodyUpdateTradeClose,
    EnterLongShortPairBody,
    ExitLongShortPairBody,
    MarginAccountNewOrderResponseFULL,
)
from constants import TradeDirection
from schema.longshorttrade import (
    LongShortTradeQuery,
)
from common_python.pred_serv_models.longshortpair import LongShortPairQuery
from schema.cloudlog import (
    slack_log_close_ls_trade_notif,
    slack_log_close_trade_notif,
    slack_log_enter_trade_notif,
)
from schema.trade import Trade, TradeQuery
from common_python.pred_serv_models.strategy import Strategy, StrategyQuery
from log import LogExceptionContext, get_logger

from math_utils import (
    calc_long_trade_net_result,
    calc_long_trade_perc_result,
    calc_short_trade_net_result,
    calc_short_trade_perc_result,
    safe_divide,
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
        short_order = req_body.short_side_order
        long_order = req_body.long_side_order

        buy_open_price = safe_divide(
            float(long_order.cummulativeQuoteQty),
            float(long_order.executedQty),
            0.0,
        )

        sell_open_price = safe_divide(
            float(short_order.cummulativeQuoteQty),
            float(short_order.executedQty),
            0.0,
        )

        sell_side_trade_body = {
            "open_time_ms": short_order.transactTime,
            "quantity": short_order.executedQty,
            "open_price": safe_divide(
                float(short_order.cummulativeQuoteQty),
                float(short_order.executedQty),
                0.0,
            ),
            "symbol": short_order.symbol,
            "direction": TradeDirection.SHORT,
        }
        sell_side_trade_id = TradeQuery.create_entry(sell_side_trade_body)

        buy_side_trade_body = {
            "open_time_ms": long_order.transactTime,
            "quantity": long_order.executedQty,
            "open_price": safe_divide(
                float(long_order.cummulativeQuoteQty),
                float(long_order.executedQty),
                0.0,
            ),
            "symbol": long_order.symbol,
            "direction": TradeDirection.LONG,
        }
        buy_side_trade_id = TradeQuery.create_entry(buy_side_trade_body)

        slack_log_enter_trade_notif(BodyCreateTrade(**sell_side_trade_body))
        slack_log_enter_trade_notif(BodyCreateTrade(**buy_side_trade_body))

        LongShortPairQuery.update_entry(
            pair.id,
            {
                "buy_open_price": buy_open_price,
                "sell_open_price": sell_open_price,
                "buy_open_qty_in_base": float(long_order.executedQty),
                "buy_open_qty_in_quote": float(long_order.cummulativeQuoteQty),
                "sell_open_qty_in_quote": float(short_order.cummulativeQuoteQty),
                "debt_open_qty_in_base": float(short_order.executedQty),
                "buy_side_trade_id": buy_side_trade_id,
                "buy_open_time_ms": long_order.transactTime,
                "sell_open_time_ms": short_order.transactTime,
                "sell_side_trade_id": sell_side_trade_id,
                "in_position": True,
                "is_no_loan_available_err": False,
            },
        )


def get_trade_net_result(
    open_qty: float, order: MarginAccountNewOrderResponseFULL, is_short: bool
):
    if is_short is True:
        return open_qty - float(order.cummulativeQuoteQty)
    else:
        return float(order.cummulativeQuoteQty) - open_qty


def get_trade_perc_result(net_result: float, size: float):
    return (safe_divide(net_result, size, 0)) * 100


def update_trade_close(
    trade_id: int,
    open_qty: float,
    order: MarginAccountNewOrderResponseFULL,
    is_short: bool,
):
    if is_short:
        net_result = open_qty - float(order.cummulativeQuoteQty)
    else:
        net_result = float(order.cummulativeQuoteQty) - open_qty

    perc_result = (safe_divide(net_result, open_qty, 0)) * 100

    TradeQuery.update_trade(
        trade_id,
        {
            "close_price": float(order.price),
            "close_time_ms": order.transactTime,
            "net_result": net_result,
            "percent_result": perc_result,
        },
    )


def exit_longshort_trade(pair: LongShortPair, req_body: ExitLongShortPairBody):
    with LogExceptionContext():
        short_side_trade = TradeQuery.get_trade_by_id(pair.sell_side_trade_id)
        long_side_trade = TradeQuery.get_trade_by_id(pair.buy_side_trade_id)

        if short_side_trade is None or long_side_trade is None:
            return

        long_side_order = req_body.long_side_order
        short_side_order = req_body.short_side_order

        update_trade_close(
            short_side_trade.id, pair.sell_open_qty_in_quote, short_side_order, True
        )
        update_trade_close(
            long_side_trade.id, pair.buy_open_qty_in_quote, long_side_order, False
        )

        long_side_net_result = get_trade_net_result(
            pair.buy_open_qty_in_quote, long_side_order, False
        )
        short_side_net_result = get_trade_net_result(
            pair.sell_open_qty_in_quote, short_side_order, True
        )

        combined_result = long_side_net_result + short_side_net_result
        combined_perc_result = (
            safe_divide(
                combined_result,
                pair.buy_open_qty_in_quote + pair.sell_open_qty_in_quote,
                0,
            )
        ) * 100

        LongShortTradeQuery.create_entry(
            {
                "long_short_group_id": pair.long_short_group_id,
                "long_side_trade_id": long_side_trade.id,
                "short_side_trade_id": short_side_trade.id,
                "net_result": combined_result,
                "percent_result": combined_perc_result,
                "close_time_ms": req_body.short_side_order.transactTime,
                "open_time_ms": pair.buy_open_time_ms,
            }
        )
        LongShortPairQuery.delete_entry(pair.id)

        slack_log_close_ls_trade_notif(
            {
                "long_side_symbol": pair.buy_symbol,
                "short_side_symbol": pair.sell_symbol,
                "net_result": combined_result,
                "combined_perc_result": combined_perc_result,
                "close_time_ms": req_body.short_side_order.transactTime,
                "open_time_ms": pair.buy_open_time_ms,
                "long_side_net_result": long_side_net_result,
                "short_side_net_result": short_side_net_result,
                "long_side_perc_result": get_trade_perc_result(
                    long_side_net_result, pair.buy_open_qty_in_quote
                ),
                "short_side_perc_result": get_trade_perc_result(
                    short_side_net_result, pair.sell_open_qty_in_quote
                ),
            }
        )
