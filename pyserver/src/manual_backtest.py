from typing import Dict, List
from code_gen_template import BACKTEST_MANUAL_TEMPLATE
from dataset import read_dataset_to_mem
from model_backtest import Positions
from query_dataset import DatasetQuery
from request_types import BodyCreateManualBacktest


START_BALANCE = 10000
FEES_PERC = 0.1
SLIPPAGE_PERC = 0.001


def run_manual_backtest(backtestInfo: BodyCreateManualBacktest):
    dataset = DatasetQuery.fetch_dataset_by_id(backtestInfo.dataset_id)
    dataset_df = read_dataset_to_mem(dataset.dataset_name)

    replacements = {
        "{ENTER_TRADE_FUNC}": backtestInfo.enter_trade_cond,
        "{EXIT_TRADE_FUNC}": backtestInfo.exit_trade_cond,
    }

    backtest = ManualBacktest(
        START_BALANCE,
        FEES_PERC,
        SLIPPAGE_PERC,
        replacements,
        backtestInfo.use_short_selling,
    )

    assert dataset.timeseries_column is not None
    assert dataset.price_column is not None

    for _, row in dataset_df.iterrows():
        backtest.process_df_row(row, dataset.price_column, dataset.timeseries_column)

    print(backtest)


class ManualBacktest:
    def __init__(
        self,
        start_balance: float,
        fees_perc: float,
        slippage_perc: float,
        enter_and_exit_criteria_placeholders: Dict,
        use_short_selling: bool,
    ) -> None:
        self.enter_and_exit_criteria_placeholders = enter_and_exit_criteria_placeholders
        self.positions = Positions(
            start_balance, 1 - (fees_perc / 100), 1 - (slippage_perc / 100)
        )
        self.history: List = []
        self.use_short_selling = use_short_selling

    def process_df_row(self, df_row, price_col, timeseries_col):
        code = BACKTEST_MANUAL_TEMPLATE
        for key, value in self.enter_and_exit_criteria_placeholders.items():
            code = code.replace(key, str(value))

        results_dict = {"df_row": df_row}
        exec(code, globals(), results_dict)

        is_enter_trade = results_dict["is_enter_trade"]
        is_exit_trade = results_dict["is_exit_trade"]

        kline_open_time = df_row[timeseries_col]
        price = df_row[price_col]

        self.tick(price, kline_open_time, is_enter_trade, is_exit_trade)

    def update_data(self, price: float, kline_open_time: int):
        self.positions.update_balance(price, 0, kline_open_time)

    def tick(
        self,
        price: float,
        kline_open_time: int,
        should_long: bool,
        should_short: bool,
    ):
        self.update_data(price, kline_open_time)

        if self.positions.position > 0 and should_long is False:
            self.positions.close_long(price, kline_open_time)

        if (
            self.positions.short_debt > 0
            and should_short is False
            and self.use_short_selling is True
        ):
            self.positions.close_short(price, kline_open_time)

        if self.positions.cash > 0 and should_long:
            self.positions.go_long(price, 0, kline_open_time)

        if (
            self.positions.short_debt == 0
            and should_short is True
            and self.use_short_selling is True
        ):
            self.positions.go_short(price, 0, kline_open_time)
