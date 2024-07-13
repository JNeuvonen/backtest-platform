from common_python.pred_serv_models.strategy_group import StrategyGroupQuery
from common_python.pred_serv_models.strategy import StrategyQuery


def transform_db():
    strategy_groups = StrategyGroupQuery.get_all()
    strategies = StrategyQuery.get_strategies()

    groups_processed = 0

    for item in strategy_groups:
        StrategyGroupQuery.update(item.id, {"force_num_required_klines": True})

        print(f"Groups processed {groups_processed}")
        groups_processed += 1

    strategies_processed = 0
    for item in strategies:
        StrategyQuery.update_strategy(item.id, {"force_num_required_klines": True})

        print(f"Strategies processed {strategies_processed}")
        strategies_processed += 1


transform_db()
