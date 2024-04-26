from typing import List
from constants import BINANCE_BACKTEST_PRICE_COL, AppConstants
from dataset import get_row_count, read_columns_to_mem, read_dataset_to_mem
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


def find_pair_candidates(
    kline_open_time: int, backtest_info: BodyCreateLongShortBacktest
):
    with LogExceptionContext():
        pass


async def run_long_short_backtest(backtest_info: BodyCreateLongShortBacktest):
    with LogExceptionContext():
        longest_dataset_id = get_longest_dataset_id(backtest_info)
        for item in backtest_info.datasets:
            dataset = DatasetQuery.fetch_dataset_by_id(item)

            dataset_name = dataset.dataset_name
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

        kline_open_times = read_columns_to_mem(
            AppConstants.DB_DATASETS,
            longest_dataset.dataset_name,
            [longest_dataset.timeseries_column],
        )

        candles_time_delta = get_df_candle_size(
            kline_open_times, dataset.timeseries_column, formatted=False
        )

        long_short_backtest = LongShortOnUniverseBacktest(
            backtest_info, candles_time_delta
        )

        if kline_open_times is None:
            raise Exception("Kline_open_times df was none.")

        for _, row in kline_open_times.iterrows():
            print(row)


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
