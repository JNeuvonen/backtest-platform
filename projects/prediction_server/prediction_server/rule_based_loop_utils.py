from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional
from datetime import datetime
from common_python.array import rm_items_from_list
from common_python.binance.funcs import (
    get_top_coins_by_usdt_volume,
    get_trade_quantities_precision,
)

from common_python.pred_serv_models.refetch_strategy_signal import (
    RefetchStrategySignalQuery,
)
from common_python.pred_serv_models.trade_info_tick import TradeInfoTickQuery
from common_python.binance.funcs import infer_assets
from prediction_server.log import LogExceptionContext

from prediction_server.utils import get_current_timestamp_ms
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


def get_strategy_name(group_name: str, symbol: str):
    return f"{group_name}_{symbol}".upper()


def refresh_adaptive_strategy_group(strategy_group: StrategyGroup):
    old_strategies = StrategyQuery.get_strategies_by_strategy_group_id(
        strategy_group.id
    )

    top_coins_by_usdt_vol = get_top_coins_by_usdt_volume(
        strategy_group.num_symbols_for_auto_adaptive
    )

    strategies_to_be_deleted = []
    found_symbols = []

    for item in old_strategies:
        if item.symbol not in top_coins_by_usdt_vol and item.is_in_position is False:
            strategies_to_be_deleted.append(item.id)
        else:
            found_symbols.append(item.symbol)

    top_coins_by_usdt_vol = rm_items_from_list(top_coins_by_usdt_vol, found_symbols)
    precisions = get_trade_quantities_precision(top_coins_by_usdt_vol)

    strategy_symbols_info = []

    for item in top_coins_by_usdt_vol:
        assets = infer_assets(item)
        precision = precisions[item]

        strategy_symbols_info.append(
            {
                "name": get_strategy_name(strategy_group.name, item),
                "strategy_group_id": strategy_group.id,
                "symbol": item,
                "base_asset": assets["baseAsset"],
                "quote_asset": assets["quoteAsset"],
                "trade_quantity_precision": precision,
                "enter_trade_code": strategy_group.enter_trade_code,
                "exit_trade_code": strategy_group.exit_trade_code,
                "fetch_datasources_code": strategy_group.fetch_datasources_code,
                "candle_interval": strategy_group.candle_interval,
                "priority": strategy_group.priority,
                "num_req_klines": strategy_group.num_req_klines,
                "kline_size_ms": strategy_group.kline_size_ms,
                "minimum_time_between_trades_ms": strategy_group.minimum_time_between_trades_ms,
                "maximum_klines_hold_time": strategy_group.maximum_klines_hold_time,
                "allocated_size_perc": strategy_group.allocated_size_perc,
                "take_profit_threshold_perc": strategy_group.take_profit_threshold_perc,
                "stop_loss_threshold_perc": strategy_group.stop_loss_threshold_perc,
                "use_time_based_close": strategy_group.use_time_based_close,
                "use_profit_based_close": strategy_group.use_profit_based_close,
                "use_stop_loss_based_close": strategy_group.use_stop_loss_based_close,
                "use_taker_order": strategy_group.use_taker_order,
                "should_calc_stops_on_pred_serv": strategy_group.should_calc_stops_on_pred_serv,
                "is_leverage_allowed": strategy_group.is_leverage_allowed,
                "is_short_selling_strategy": strategy_group.is_short_selling_strategy,
            }
        )

    if len(strategies_to_be_deleted) > 0:
        StrategyQuery.delete_strategies_by_ids(strategies_to_be_deleted)

    if len(strategy_symbols_info) > 0:
        StrategyQuery.create_many(strategy_symbols_info)

    StrategyGroupQuery.update_last_adaptive_group_recalc(strategy_group.id)


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

    def update_strategy_groups(self):
        for item in self.strategy_groups:
            if item.is_auto_adaptive_group is False:
                continue

            now = datetime.now()
            difference = now - item.last_adaptive_group_recalc

            if difference.days >= item.num_days_for_group_recalc:
                refresh_adaptive_strategy_group(item)

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
