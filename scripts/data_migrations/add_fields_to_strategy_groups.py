from common_python.pred_serv_models.strategy_group import StrategyGroupQuery
from common_python.pred_serv_models.strategy import Strategy, StrategyQuery


def transform_db():
    strategy_by_group: List[Strategy] = StrategyQuery.get_one_strategy_per_group()

    for item in strategy_by_group:
        update_dict = {
            "is_auto_adaptive_group": True,
            "num_symbols_for_auto_adaptive": 25,
            "num_days_for_group_recalc": 7,
            "enter_trade_code": item.enter_trade_code,
            "exit_trade_code": item.exit_trade_code,
            "fetch_datasources_code": item.fetch_datasources_code,
            "candle_interval": item.candle_interval,
            "priority": item.priority,
            "num_req_klines": item.num_req_klines,
            "kline_size_ms": item.kline_size_ms,
            "minimum_time_between_trades_ms": item.minimum_time_between_trades_ms,
            "maximum_klines_hold_time": item.maximum_klines_hold_time,
            "allocated_size_perc": item.allocated_size_perc,
            "take_profit_threshold_perc": item.take_profit_threshold_perc,
            "stop_loss_threshold_perc": item.stop_loss_threshold_perc,
            "use_time_based_close": item.use_time_based_close,
            "use_profit_based_close": item.use_profit_based_close,
            "use_stop_loss_based_close": item.use_stop_loss_based_close,
            "use_taker_order": item.use_taker_order,
            "should_calc_stops_on_pred_serv": item.should_calc_stops_on_pred_serv,
            "is_leverage_allowed": item.is_leverage_allowed,
            "is_short_selling_strategy": item.is_short_selling_strategy,
        }
        StrategyGroupQuery.update(item.strategy_group_id, update_dict)
        StrategyGroupQuery.update_last_adaptive_group_recalc(item.strategy_group_id)


transform_db()
