from typing import List
from common_python.pred_serv_models.data_transformation import DataTransformationQuery
from common_python.pred_serv_models.strategy import StrategyQuery
from common_python.pred_serv_models.strategy_group import StrategyGroupQuery
from common_python.pred_serv_orm import StrategyGroup


def transform_db():
    strategy_by_group = StrategyQuery.get_one_strategy_per_group()

    data_transformations_dict = {}

    for strategy in strategy_by_group:
        transformations = DataTransformationQuery.get_transformations_by_strategy(
            strategy.id
        )

        data_transformations_dict[strategy.strategy_group] = transformations

    preserved_ids: List[int] = []

    for group_name, transformations in data_transformations_dict.items():
        transformation_ids = [item.id for item in transformations]

        for item in transformation_ids:
            preserved_ids.append(item)

        strategy_group_id = StrategyGroupQuery.create_entry(
            {
                "name": group_name,
                "transformation_ids": transformation_ids,
                "is_disabled": False,
            }
        )

        for transformation_id in transformation_ids:
            DataTransformationQuery.update(
                transformation_id, {"strategy_group_id": strategy_group_id}
            )

        StrategyQuery.update_strategy_group_id(group_name, strategy_group_id)


transform_db()
