import json
from typing import Dict, List

from log import LogExceptionContext
from query_trade import TradeQuery
from query_weights import ModelWeights
from query_trainjob import TrainJob, TrainJobQuery
from query_backtest import BacktestQuery
from request_types import BodyRunBacktest
from code_gen_template import BACKTEST_TEMPLATE


class Direction:
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
        self.balance_on_start_of_trade = start_balance
        self.fees_multiplier = 1 - (fees / 100)
        self.slippage_multiplier = 1 - (slippage / 100)
        self.enter_and_exit_criteria_placeholders = enter_and_exit_criteria_placeholders
        self.position = Direction.CASH
        self.price_on_start_of_trade: float
        self.last_price: float | None = None
        self.trade_predictions: List[float] = []
        self.trade_prices: List[float] = []
        self.time_on_start_of_trade: int
        self.trades: List[dict] = []
        self.balance_history: List[dict] = []

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

        if should_enter_trade is False and self.position == Direction.LONG:
            self.close_long(price, prediction, kline_open_time)

        if should_exit_trade is False and self.position == Direction.SHORT:
            self.close_short(price, prediction, kline_open_time)

        if should_enter_trade is True and self.position == Direction.CASH:
            self.balance_on_start_of_trade = self.balance
            self.price_on_start_of_trade = price
            self.time_on_start_of_trade = kline_open_time
            self.position = Direction.LONG

        if should_exit_trade is True and self.position == Direction.CASH:
            self.balance_on_start_of_trade = self.balance
            self.price_on_start_of_trade = price
            self.time_on_start_of_trade = kline_open_time
            self.position = Direction.SHORT

        self.update_data(price, kline_open_time, prediction)

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

        if self.position == Direction.LONG:
            profit = (price - self.last_price) * (self.balance / self.last_price)
            self.balance += profit

        if self.position == Direction.SHORT:
            profit = (self.last_price - price) * (self.balance / self.last_price)
            self.balance += profit

        if self.position == Direction.LONG or self.position == Direction.SHORT:
            self.trade_predictions.append(prediction)
            self.trade_prices.append(price)

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

    def close_long(self, price: float, prediction: float, trade_close_time: int):
        if self.position == Direction.LONG:
            profit = (price - self.last_price) * (self.balance / self.last_price)
            self.balance += profit
        balance_before = self.balance_on_start_of_trade
        finished_trade = {
            "open_price": self.price_on_start_of_trade,
            "start_time": self.time_on_start_of_trade,
            "direction": Direction.LONG,
            "end_price": price,
            "end_time": trade_close_time,
            "net_result": self.balance - balance_before,
            "percent_result": (self.balance / balance_before - 1) * 100,
            "predictions": self.trade_predictions,
            "prices": self.trade_prices,
        }
        self.trade_prices.append(price)
        self.trade_predictions.append(prediction)
        self.trades.append(finished_trade)
        self.position = Direction.CASH

    def close_short(self, price: float, prediction: float, trade_close_time: int):
        if self.position == Direction.SHORT:
            profit = (self.last_price - price) * (self.balance / self.last_price)
            self.balance += profit

        balance_before = self.balance_on_start_of_trade
        finished_trade = {
            "open_price": self.price_on_start_of_trade,
            "start_time": self.time_on_start_of_trade,
            "direction": Direction.SHORT,
            "end_price": price,
            "end_time": trade_close_time,
            "net_result": self.balance - balance_before,
            "percent_result": (self.balance / balance_before - 1) * 100,
            "predictions": self.trade_predictions,
            "prices": self.trade_prices,
        }
        self.trade_prices.append(price)
        self.trade_predictions.append(prediction)
        self.trades.append(finished_trade)
        self.position = Direction.CASH


def run_backtest(train_job_id: int, backtestInfo: BodyRunBacktest):
    with LogExceptionContext():
        train_job_detailed = TrainJobQuery.get_train_job_detailed(train_job_id)
        train_job: TrainJob = train_job_detailed["train_job"]
        epochs: List[ModelWeights] = train_job_detailed["epochs"]

        prices = json.loads(train_job.backtest_prices)
        kline_open_time = json.loads(train_job.backtest_kline_open_times)
        predictions = json.loads(epochs[backtestInfo.epoch_nr]["val_predictions"])

        assert len(prices) == len(predictions)

        replacements = {
            "{ENTER_AND_EXIT_CRITERIA_FUNCS}": backtestInfo.enter_and_exit_criteria,
            "{PREDICTION}": None,
        }

        backtest = Backtest(10000, 0.1, 0.1, replacements)

        for idx in range(len(prices)):
            price = prices[idx]
            prediction = predictions[idx]
            backtest.tick(price, prediction[0], kline_open_time[idx])

        backtest_id = BacktestQuery.create_backtest_entry(
            backtestInfo.enter_and_exit_criteria,
            backtest.balance_history,
            epochs[backtestInfo.epoch_nr]["id"],
        )

        TradeQuery.create_many_trade_entry(backtest_id, backtest.trades)

        return {"data": backtest.balance_history, "trades": backtest.trades}


class Positions:
    def __init__(self, start_balance, fees, slippage):
        self.cash = start_balance
        self.fees = fees
        self.slippage = slippage
        self.position = 0.0
        self.short_debt = 0.0
        self.balance_history = []

    def close_long(self, price: float):
        self.cash += price * self.position
        self.position = 0.0

    def close_short(self, price: float):
        price_to_close = price * self.short_debt
        self.cash -= price_to_close
        self.short_debt = 0.0

    def go_long(self, price: float):
        self.position = self.cash / price
        self.cash = 0.0

    def go_short(self, price: float):
        short_proceedings = self.cash * price
        self.short_debt = self.cash / price
        self.cash += short_proceedings

    def update_balance(self, price: float):
        portfolio_worth = self.cash

        if self.position > 0.0:
            portfolio_worth += price * self.position

        if self.short_debt > 0.0:
            portfolio_worth -= price * self.short_debt
        self.balance_history.append(portfolio_worth)


class BacktestV2:
    def __init__(
        self,
        start_balance: float,
        fees: float,
        slippage: float,
        enter_and_exit_criteria_placeholders: Dict,
    ) -> None:
        self.balance = start_balance
        self.enter_and_exit_criteria_placeholders = enter_and_exit_criteria_placeholders
        self.positions = Positions(
            start_balance, (1 - fees) / 100, 1 - (slippage / 100)
        )

    def enter_kline(self, price: float, prediction: float, kline_open_time: int):
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
        self.tick(
            price, prediction, kline_open_time, should_enter_trade, should_exit_trade
        )

    def tick(
        self,
        price: float,
        prediction: float,
        kline_open_time: int,
        should_long: bool,
        should_short: bool,
    ):
        if self.positions.position > 0 and should_long is False:
            self.positions.close_long(price)

        if self.positions.position < 0 and should_short is False:
            self.positions.sell(price)
