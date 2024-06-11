from common_python.pred_serv_models.trade_info_tick import TradeInfoTick
from common_python.pred_serv_models.trade import TradeQuery
from common_python.pred_serv_models.strategy import StrategyQuery


def find_by_strat_id(strategy_id, strategies):
    for item in strategies:
        if item.id == strategy_id:
            return item
    return None


def transform_db():
    trades = TradeQuery.get_trades()
    strategies = StrategyQuery.get_strategies()

    for item in trades:
        if item.strategy_id is not None:
            strategy = find_by_strat_id(item.strategy_id, strategies)

            TradeQuery.update(
                item.id, {"strategy_group_id": strategy.strategy_group_id}
            )


transform_db()
