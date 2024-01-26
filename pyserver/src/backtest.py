from typing import Dict, List

from pandas.io.parquet import json
from log import LogExceptionContext
from orm import Dataset, Model, ModelWeights, TrainJob, TrainJobQuery
from request_types import BodyRunBacktest
from code_gen_template import BACKTEST_TEMPLATE


class Position:
    LONG = "long"
    SHORT = "short"
    CASH = "cash"


class Backtest:
    def __init__(
        self,
        start_balance: float,
        fees: float,
        slippage: float,
        enter_and_exit_criteria_placeholders: Dict,
    ) -> None:
        self.balance = start_balance
        self.fees = fees / 100
        self.slippage = slippage / 100
        self.enter_and_exit_criteria_placeholders = enter_and_exit_criteria_placeholders
        self.position = Position.CASH
        self.price_on_start_of_trade: float = 100

    def tick(self, price: float, prediction: float):
        code = BACKTEST_TEMPLATE
        for key, value in self.enter_and_exit_criteria_placeholders.items():
            if key == "{PREDICTION}":
                code = code.replace(key, str(prediction))
            else:
                code = code.replace(key, str(value))

        results_dict: Dict[str, bool] = {}
        exec(code, globals(), results_dict)

        should_enter_trade = results_dict["enter_trade"]
        should_exit_trade = results_dict["exit_trade"]

        if should_enter_trade is False and self.position == Position.LONG:
            self.close_long(price)

        if should_exit_trade is False and self.position == Position.SHORT:
            self.close_short(price)

        if should_enter_trade is True and self.position == Position.CASH:
            self.price_on_start_of_trade = price
            self.position = Position.LONG

        if should_exit_trade is True and self.position == Position.CASH:
            self.price_on_start_of_trade = price
            self.position = Position.SHORT

    def init_long(self, price: float):
        self.price_on_start_of_trade = price

    def close_long(self, price: float):
        self.balance = (
            price
            / self.price_on_start_of_trade
            * self.fees
            * self.fees
            * self.slippage
            * self.slippage
        )
        self.position = Position.CASH

    def close_short(self, price: float):
        if price > self.price_on_start_of_trade:
            self.balance += (
                (price / self.price_on_start_of_trade - 1) * -1 * self.balance
            )
        else:
            self.balance += abs(price / self.price_on_start_of_trade - 1) * self.balance


def run_backtest(train_job_id: int, backtestInfo: BodyRunBacktest):
    with LogExceptionContext():
        train_job_detailed = TrainJobQuery.get_train_job_detailed(train_job_id)
        model: Model = train_job_detailed["model"]
        train_job: TrainJob = train_job_detailed["train_job"]
        epochs: List[ModelWeights] = train_job_detailed["epochs"]
        dataset: Dataset = train_job_detailed["dataset"]

        target_col = json.loads(train_job.validation_target_before_scale)
        predictions = json.loads(epochs[backtestInfo.epoch_nr]["val_predictions"])

        assert len(target_col) == len(predictions)

        replacements = {
            "{ENTER_AND_EXIT_CRITERIA_FUNCS}": backtestInfo.enter_and_exit_criteria,
            "{PREDICTION}": None,
        }

        backtest = Backtest(10000, 0.1, 0.1, replacements)

        for idx in range(len(target_col)):
            price = target_col[idx]
            prediction = predictions[idx]
            backtest.tick(price, prediction[0])

        print(backtest)
