from os import times
from time import time
from typing import Dict, List, Set
from code_gen_template import BACKTEST_LONG_SHORT_BUYS_AND_SELLS
from constants import BINANCE_BACKTEST_PRICE_COL, AppConstants
from dataset import (
    get_row_count,
    read_all_cols_matching_kline_open_times,
    read_columns_to_mem,
    read_dataset_to_mem,
)
from db import exec_python, get_df_candle_size
from log import LogExceptionContext
from query_data_transformation import DataTransformationQuery
from query_dataset import DatasetQuery
from request_types import BodyCreateLongShortBacktest
from utils import PythonCode

START_BALANCE = 10000


def get_longest_dataset_id(backtest_info: BodyCreateLongShortBacktest):
    longest_id = None
    longest_row_count = -1000
    for item in backtest_info.datasets:
        dataset = DatasetQuery.fetch_dataset_by_id(item)
        row_count = get_row_count(dataset.dataset_name)

        if longest_row_count < row_count:
            longest_id = dataset.id
            longest_row_count = row_count
    return longest_id


def get_symbol_buy_and_sell_decision(df_row, replacements: Dict):
    code = BACKTEST_LONG_SHORT_BUYS_AND_SELLS

    for key, value in replacements.items():
        code = code.replace(key, str(value))

    results_dict = {"df_row": df_row}
    exec(code, globals(), results_dict)

    ret = {
        "is_valid_buy": results_dict["is_valid_buy"],
        "is_valid_sell": results_dict["is_valid_sell"],
    }
    return ret


def find_pair_candidates(
    kline_open_time: int,
    backtest_info: BodyCreateLongShortBacktest,
    dataset_id_to_name_map: Dict,
    dataset_id_to_ts_col: Dict,
):
    sell_candidates: Set = set()
    buy_candidates: Set = set()

    exec_py_replacements = {
        "{BUY_COND_FUNC}": backtest_info.buy_cond,
        "{SELL_COND_FUNC}": backtest_info.sell_cond,
    }

    with LogExceptionContext():
        for item in backtest_info.datasets:
            dataset_name = dataset_id_to_name_map[str(item)]
            timeseries_col = dataset_id_to_ts_col[str(item)]
            df = read_all_cols_matching_kline_open_times(
                dataset_name, timeseries_col, [kline_open_time]
            )
            df_row = df.iloc[0]
            buy_and_sell_decisions = get_symbol_buy_and_sell_decision(
                df_row, exec_py_replacements
            )

            is_valid_buy = buy_and_sell_decisions["is_valid_buy"]
            is_valid_sell = buy_and_sell_decisions["is_valid_sell"]

            if is_valid_buy is True:
                buy_candidates.add(item)

            if is_valid_sell is True:
                sell_candidates.add(item)

    return {"sell_candidates": sell_candidates, "buy_candidates": buy_candidates}


async def run_long_short_backtest(backtest_info: BodyCreateLongShortBacktest):
    with LogExceptionContext():
        longest_dataset_id = get_longest_dataset_id(backtest_info)
        dataset_id_to_name_map = {}
        dataset_id_to_timeseries_col_map = {}
        for item in backtest_info.datasets:
            dataset = DatasetQuery.fetch_dataset_by_id(item)

            dataset_name = dataset.dataset_name
            dataset_id_to_name_map[str(item)] = dataset_name
            dataset_id_to_timeseries_col_map[str(item)] = dataset.timeseries_column
            DatasetQuery.update_price_column(dataset_name, BINANCE_BACKTEST_PRICE_COL)

            for data_transform_id in backtest_info.data_transformations:
                transformation = DataTransformationQuery.get_transformation_by_id(
                    data_transform_id
                )
                python_program = PythonCode.on_dataset(
                    dataset_name, transformation.transformation_code
                )
                exec_python(python_program)

        if longest_dataset_id is None:
            raise Exception("Longest dataset_id was none.")

        longest_dataset = DatasetQuery.fetch_dataset_by_id(longest_dataset_id)
        timeseries_col = longest_dataset.timeseries_column

        kline_open_times = read_columns_to_mem(
            AppConstants.DB_DATASETS,
            longest_dataset.dataset_name,
            [timeseries_col],
        )

        candles_time_delta = get_df_candle_size(
            kline_open_times, timeseries_col, formatted=False
        )

        long_short_backtest = LongShortOnUniverseBacktest(
            backtest_info, candles_time_delta
        )

        if kline_open_times is None:
            raise Exception("Kline_open_times df was none.")

        for _, row in kline_open_times.iterrows():
            pair_candidates = find_pair_candidates(
                row[timeseries_col],
                backtest_info,
                dataset_id_to_name_map,
                dataset_id_to_timeseries_col_map,
            )

            kline_open_time = row[timeseries_col]
            long_short_backtest.process_bar(
                kline_open_time=kline_open_time, buy_and_sell_candidates=pair_candidates
            )


class BacktestRules:
    def __init__(self, backtest_details: BodyCreateLongShortBacktest):
        self.buy_cond = backtest_details.buy_cond
        self.sell_cond = backtest_details.sell_cond

        self.use_profit_based_close = backtest_details.use_profit_based_close
        self.use_stop_loss_based_close = backtest_details.use_stop_loss_based_close
        self.use_time_based_close = backtest_details.use_time_based_close

        self.max_klines_until_close = backtest_details.klines_until_close
        self.take_profit_threshold_perc = backtest_details.take_profit_threshold_perc
        self.stop_loss_threshold_perc = backtest_details.stop_loss_threshold_perc


class BacktestStats:
    def __init__(self, candles_time_delta: int):
        self.cumulative_time = 0
        self.candles_time_delta = candles_time_delta


class USDTPosition:
    def __init__(self, balance: int):
        self.balance = balance


class PairTrade:
    def __init__(
        self,
        buy_id: int,
        sell_id: int,
        sell_amount: float,
        buy_amount: float,
        trade_open_time,
        sell_price: float,
        buy_price: float,
    ):
        self.buy_id = buy_id
        self.sell_id = sell_id
        self.sell_amount = sell_amount
        self.buy_amount = buy_amount
        self.trade_open_time = trade_open_time
        self.sell_price = sell_price
        self.buy_price = buy_price


class LongShortOnUniverseBacktest:
    def __init__(
        self, backtest_details: BodyCreateLongShortBacktest, candles_time_delta: int
    ):
        self.rules = BacktestRules(backtest_details)
        self.stats = BacktestStats(candles_time_delta)
        self.usdt_position = USDTPosition(START_BALANCE)

        self.history: List = []
        self.active_pairs: List[PairTrade] = []
        self.completed_trades: List = []

    def process_bar(self, kline_open_time: int, buy_and_sell_candidates: Dict):
        print("helo world")
        pass
