from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

from common_python.pred_serv_models.refetch_strategy_signal import (
    RefetchStrategySignalQuery,
)
from common_python.pred_serv_models.trade_info_tick import TradeInfoTickQuery
from log import LogExceptionContext
from utils import get_current_timestamp_ms
import pandas as pd
from common_python.pred_serv_models.data_transformation import (
    DataTransformation,
    DataTransformationQuery,
)
from common_python.pred_serv_models.trade import Trade, TradeQuery
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
        active_trade: Trade,
        klines: pd.DataFrame = pd.DataFrame(),
        last_kline_open_time_sec: Optional[int] = None,
    ):
        self.name = name
        self.transformations = transformations
        self.strategy_group = strategy_group
        self.dataset = klines
        self.strategy = strategy
        self.last_kline_open_time_sec = last_kline_open_time_sec
        self.active_trade = active_trade
        self.new_data_arrived = False

        self.should_enter_trade = strategy.should_enter_trade
        self.should_close_trade = strategy.should_close_trade
        self.is_on_pred_serv_err = strategy.is_on_pred_serv_err
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


def find_active_trade(strategy: Strategy, active_trades: List[Trade]):
    for item in active_trades:
        if item.strategy_id == strategy.id:
            return item
    return None


class RuleBasedLoopManager:
    def __init__(self) -> None:
        strategies = StrategyQuery.get_active_strategies()

        self.datasets: Dict[str, LocalDataset] = {}

        self.active_trades = TradeQuery.get_open_trades()
        self.strategies = strategies
        self.strategy_groups = StrategyGroupQuery.get_all_active()
        self.data_transformations = (
            DataTransformationQuery.get_all_strategy_group_transformations()
        )
        self.last_refetch = get_current_timestamp_ms()
        self.db_refetch_interval = 60000 * 15
        self.new_trade_info_ticks = []
        self.system_start_time_ms = get_current_timestamp_ms()

        for item in strategies:
            self.init_local_dataset(item)

    def init_local_dataset(self, item: Strategy):
        group = find_strategy_group_by_id(self.strategy_groups, item.strategy_group_id)

        if group is None:
            return

        transformations = find_transformations_by_group(
            group, self.data_transformations
        )

        active_trade = find_active_trade(item, self.active_trades)

        if active_trade is not None:
            pass

        prev_local_dataset = self.datasets.get(item.name)

        if prev_local_dataset is not None:
            local_dataset = LocalDataset(
                item.name,
                transformations,
                group,
                item,
                active_trade,
                prev_local_dataset.dataset,
                prev_local_dataset.last_kline_open_time_sec,
            )
        else:
            local_dataset = LocalDataset(
                item.name, transformations, group, item, active_trade
            )

        self.datasets[item.name] = local_dataset

    def create_new_dataset_objs(self):
        for item in self.strategies:
            self.init_local_dataset(item)

    def update_local_state(self):
        with LogExceptionContext(re_raise=False):
            current_time = get_current_timestamp_ms()
            if current_time >= self.last_refetch + self.db_refetch_interval:
                self.last_refetch = current_time
                self.strategies = StrategyQuery.get_active_strategies()
                self.active_trades = TradeQuery.get_open_trades()
                self.strategy_groups = StrategyGroupQuery.get_all_active()
                self.data_transformations = (
                    DataTransformationQuery.get_all_strategy_group_transformations()
                )

                self.create_new_dataset_objs()

    def replace_in_strategy_arr(self, strategy: Strategy):
        for idx, item in enumerate(self.strategies):
            if item.id == strategy.id:
                self.strategies[idx] = strategy
                break

    def replace_in_active_trades_arr(self, trade: Trade):
        for idx, item in enumerate(self.active_trades):
            if item.id == trade.id:
                self.active_trades[idx] = trade
                break

    def check_refetch_signal(self):
        with LogExceptionContext(re_raise=False):
            refetch_signals = RefetchStrategySignalQuery.get_all()

            strategy_ids = []
            refetch_signal_ids = []

            for item in refetch_signals:
                strategy_ids.append(item.strategy_id)
                refetch_signal_ids.append(item.id)

            updated_strategies = StrategyQuery.fetch_many_by_id(strategy_ids)

            trade_ids = []

            for item in updated_strategies:
                self.replace_in_strategy_arr(item)

                if item.is_in_position is True and item.active_trade_id is not None:
                    trade_ids.append(item.active_trade_id)

            trades = TradeQuery.fetch_many_by_id(trade_ids)

            for item in trades:
                self.replace_in_active_trades_arr(item)

            if len(refetch_signals) > 0:
                self.create_new_dataset_objs()
                RefetchStrategySignalQuery.delete_entries_by_ids(refetch_signal_ids)

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
        with LogExceptionContext(re_raise=False):
            updated_strategies_dict = {}

            for _, local_dataset in self.datasets.items():
                if local_dataset.new_data_arrived is True:
                    updated_fields = local_dataset.updated_fields
                    if updated_fields:
                        updated_strategies_dict[
                            local_dataset.strategy.id
                        ] = updated_fields.copy()
                        updated_strategies_dict[local_dataset.strategy.id][
                            "is_no_loan_available_err"
                        ] = False
                        local_dataset.updated_fields = {}
                        local_dataset.new_data_arrived = False

            if updated_strategies_dict:
                StrategyQuery.update_multiple_strategies(updated_strategies_dict)

    def add_trade_info_tick(self, price, kline_open_time_ms, trade_id):
        info_tick = {
            "price": price,
            "kline_open_time_ms": kline_open_time_ms,
            "trade_id": trade_id,
        }
        self.new_trade_info_ticks.append(info_tick)

    def create_trade_info_ticks(self):
        with LogExceptionContext(re_raise=False):
            if len(self.new_trade_info_ticks) > 0:
                TradeInfoTickQuery.create_many(self.new_trade_info_ticks)
                self.new_trade_info_ticks = []
