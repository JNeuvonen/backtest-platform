from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from utils import get_current_timestamp_ms
import pandas as pd
from common_python.pred_serv_models.data_transformation import (
    DataTransformation,
    DataTransformationQuery,
)
from common_python.pred_serv_models.strategy import Dict, Strategy, StrategyQuery
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
        strategy: Strategy,
    ):
        self.name = name
        self.transformations = transformations
        self.strategy_group = strategy_group
        self.dataset = pd.DataFrame()
        self.strategy = strategy
        self.last_kline_open_time_sec = None
        self.new_data_arrived = False

        self.should_enter_trade = False
        self.should_close_trade = False
        self.is_on_pred_serv_err = False
        self.updated_fields = {}


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

        self.datasets: Dict[str, LocalDataset] = {}

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
        local_dataset = LocalDataset(item.name, transformations, group, item)
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

    def update_local_dataset(
        self,
        strategy: Strategy,
        klines: pd.DataFrame,
        last_kline_open_time_sec: int,
        trading_state_dict: Dict,
    ):
        self.datasets[strategy.name].dataset = klines
        self.datasets[strategy.name].last_kline_open_time_sec = last_kline_open_time_sec

        current_is_on_pred_serv_err = self.datasets[strategy.name].is_on_pred_serv_err
        current_should_enter_trade = self.datasets[strategy.name].should_enter_trade
        current_should_close_trade = self.datasets[strategy.name].should_close_trade

        new_is_on_pred_serv_err = trading_state_dict["is_on_pred_serv_err"]
        new_should_enter_trade = trading_state_dict["should_enter_trade"]
        new_should_close_trade = trading_state_dict["should_close_trade"]

        self.datasets[strategy.name].updated_fields = {}

        if current_is_on_pred_serv_err != new_is_on_pred_serv_err:
            self.datasets[strategy.name].is_on_pred_serv_err = new_is_on_pred_serv_err
            self.datasets[strategy.name].updated_fields[
                "is_on_pred_serv_err"
            ] = new_is_on_pred_serv_err

        if current_should_enter_trade != new_should_enter_trade:
            self.datasets[strategy.name].should_enter_trade = new_should_enter_trade
            self.datasets[strategy.name].updated_fields[
                "should_enter_trade"
            ] = new_should_enter_trade

        if current_should_close_trade != new_should_close_trade:
            self.datasets[strategy.name].should_close_trade = new_should_close_trade
            self.datasets[strategy.name].updated_fields[
                "should_close_trade"
            ] = new_should_close_trade

        if self.datasets[strategy.name].updated_fields:
            self.datasets[strategy.name].new_data_arrived = True

    def update_changed_strategies(self):
        updated_strategies_dict = {}

        for _, local_dataset in self.datasets.items():
            if local_dataset.new_data_arrived is True:
                updated_fields = local_dataset.updated_fields
                if updated_fields:
                    updated_strategies_dict[
                        local_dataset.strategy.id
                    ] = updated_fields.copy()
                    local_dataset.updated_fields = {}
                    local_dataset.new_data_arrived = False

        if updated_strategies_dict:
            StrategyQuery.update_multiple_strategies(updated_strategies_dict)
