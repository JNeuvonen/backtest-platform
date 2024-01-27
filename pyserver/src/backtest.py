import json
from typing import Dict, List

from log import LogExceptionContext
from query_trade import TradeQuery
from query_weights import ModelWeights
from query_trainjob import TrainJob, TrainJobQuery
from query_backtest import BacktestQuery
from request_types import BodyRunBacktest
from code_gen_template import BACKTEST_TEMPLATE


class Position:
    LONG = "long"
    SHORT = "short"
    CASH = "cash"


class Trade:
    def __init__(
        self,
        start_price,
        start_time,
        direction,
        end_price,
        end_time,
        net_result,
        percent_result,
    ):
        self.open_price = start_price
        self.open_time = start_time
        self.direction = direction
        self.end_price = end_price
        self.end_time = end_time
        self.net_result = net_result
        self.percent_result = percent_result

    def to_dict(self):
        return {
            "open_price": self.open_price,
            "start_time": self.open_time,
            "direction": self.direction,
            "end_price": self.end_price,
            "end_time": self.end_time,
            "net_result": self.net_result,
            "percent_result": self.percent_result,
        }


class Backtest:
    def __init__(
        self,
        start_balance: float,
        fees: float,
        slippage: float,
        enter_and_exit_criteria_placeholders: Dict,
    ) -> None:
        self.balance = start_balance
        self.balance_on_start_of_trade = start_balance
        self.fees_multiplier = 1 - (fees / 100)
        self.slippage_multiplier = 1 - (slippage / 100)
        self.enter_and_exit_criteria_placeholders = enter_and_exit_criteria_placeholders
        self.position = Position.CASH
        self.price_on_start_of_trade: float
        self.last_price: float | None = None
        self.time_on_start_of_trade: int
        self.trades: List[Trade] = []
        self.balance_history = []

    def tick(self, price: float, prediction: float, kline_open_time: int):
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
        self.update_data(price, kline_open_time, prediction)

        if should_enter_trade is False and self.position == Position.LONG:
            self.close_long(price, kline_open_time)

        if should_exit_trade is False and self.position == Position.SHORT:
            self.close_short(price, kline_open_time)

        if should_enter_trade is True and self.position == Position.CASH:
            self.balance_on_start_of_trade = self.balance
            self.price_on_start_of_trade = price
            self.time_on_start_of_trade = kline_open_time
            self.position = Position.LONG

        if should_exit_trade is True and self.position == Position.CASH:
            self.balance_on_start_of_trade = self.balance
            self.price_on_start_of_trade = price
            self.time_on_start_of_trade = kline_open_time
            self.position = Position.SHORT

    def update_data(self, price: float, kline_open_time: int, prediction: float):
        if self.last_price is None:
            self.last_price = price
            self.balance_history.append(
                {
                    "balance": self.balance,
                    "kline_open_time": kline_open_time,
                    "price": price,
                    "prediction": prediction,
                }
            )
            return

        if self.position == Position.CASH:
            self.balance_history.append(
                {
                    "balance": self.balance,
                    "kline_open_time": kline_open_time,
                    "price": price,
                    "prediction": prediction,
                }
            )
            return

        if self.position == Position.SHORT:
            if price > self.last_price:
                self.balance += (price / self.last_price - 1) * -1 * self.balance
            else:
                self.balance += abs(price / self.last_price - 1) * self.balance

        if self.position == Position.LONG:
            self.balance = price / self.last_price * self.balance

        self.last_price = price
        self.balance_history.append(
            {
                "balance": self.balance,
                "kline_open_time": kline_open_time,
                "price": price,
                "prediction": prediction,
                "position": self.position,
            }
        )

    def close_long(self, price: float, trade_close_time: int):
        balance_before = self.balance_on_start_of_trade
        finished_trade = Trade(
            self.price_on_start_of_trade,
            self.time_on_start_of_trade,
            Position.LONG,
            price,
            trade_close_time,
            self.balance - balance_before,
            (self.balance / balance_before - 1) * 100,
        )
        self.trades.append(finished_trade)
        self.position = Position.CASH

    def close_short(self, price: float, trade_close_time: int):
        balance_before = self.balance_on_start_of_trade
        finished_trade = Trade(
            self.price_on_start_of_trade,
            self.time_on_start_of_trade,
            Position.SHORT,
            price,
            trade_close_time,
            self.balance - balance_before,
            (self.balance / balance_before - 1) * 100,
        )
        self.trades.append(finished_trade)
        self.position = Position.CASH


def run_backtest(train_job_id: int, backtestInfo: BodyRunBacktest):
    with LogExceptionContext():
        train_job_detailed = TrainJobQuery.get_train_job_detailed(train_job_id)
        train_job: TrainJob = train_job_detailed["train_job"]
        epochs: List[ModelWeights] = train_job_detailed["epochs"]

        target_col = json.loads(train_job.validation_target_before_scale)
        print(target_col)
        kline_open_time = json.loads(train_job.validation_kline_open_times)
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
            backtest.tick(price, prediction[0], kline_open_time[idx])

        backtest_id = BacktestQuery.create_backtest_entry(
            backtestInfo.enter_and_exit_criteria,
            backtest.balance_history,
            epochs[backtestInfo.epoch_nr]["id"],
        )

        trades = [item.to_dict() for item in backtest.trades]

        TradeQuery.create_many_trade_entry(backtest_id, trades)

        return {"data": backtest.balance_history, "trades": trades}
