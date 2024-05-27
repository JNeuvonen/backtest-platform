from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from utils import get_current_timestamp_ms
import pandas as pd
from common_python.pred_serv_models.data_transformation import (
    DataTransformation,
    DataTransformationQuery,
)
from common_python.pred_serv_models.strategy import Strategy, StrategyQuery
from common_python.pred_serv_models.strategy_group import (
    StrategyGroup,
    StrategyGroupQuery,
)


class LocalDataset:
    def __init__(
        self,
        name: str,
        transformations: List[DataTransformation],
        strategy_group: StrategyGroup,
    ):
        self.name = name
        self.candles_processed = 0
        self.transformations = transformations
        self.strategy_group = strategy_group
        self.dataset = pd.DataFrame()


def find_transformations_by_group(
    group: StrategyGroup, transformations: List[DataTransformation]
):
    ret = []

    for item in transformations:
        if item.id in group.transformation_ids:
            ret.append(item)
    return ret


def find_strategy_group_by_id(groups: List[StrategyGroup], id: int):
    for item in groups:
        if item.id == id:
            return item

    return None


class RuleBasedLoopManager:
    def __init__(self) -> None:
        strategies = StrategyQuery.get_active_strategies()

        self.datasets = {}

        self.strategies = strategies
        self.strategy_groups = StrategyGroupQuery.get_all_active()
        self.data_transformations = (
            DataTransformationQuery.get_all_strategy_group_transformations()
        )

        self.last_refetch = get_current_timestamp_ms()
        self.db_refetch_interval = 60000 * 60

        for item in strategies:
            self.init_local_dataset(item)

    def init_local_dataset(self, item: Strategy):
        group = find_strategy_group_by_id(self.strategy_groups, item.strategy_group_id)

        if group is None:
            return

        transformations = find_transformations_by_group(
            group, self.data_transformations
        )
        local_dataset = LocalDataset(item.name, transformations, group)
        self.datasets[item.name] = local_dataset

    def create_new_dataset_objs(self):
        for item in self.strategies:
            local_dataset = self.datasets.get(item.name)

            if local_dataset is None:
                self.init_local_dataset(item)

    def update_local_state(self):
        current_time = get_current_timestamp_ms()
        if current_time >= self.last_refetch + self.db_refetch_interval:
            self.last_refetch = current_time

            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(StrategyQuery.get_active_strategies): "strategies",
                    executor.submit(
                        StrategyGroupQuery.get_all_active
                    ): "strategy_groups",
                    executor.submit(
                        DataTransformationQuery.get_all_strategy_group_transformations
                    ): "data_transformations",
                }

                for future in as_completed(futures):
                    result = future.result()
                    attr = futures[future]
                    setattr(self, attr, result)

            self.create_new_dataset_objs()
