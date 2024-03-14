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


def run_model_backtest(train_job_id: int, backtestInfo: BodyRunBacktest):
    with LogExceptionContext():
        train_job_detailed = TrainJobQuery.get_train_job_detailed(train_job_id)
        train_job: TrainJob = train_job_detailed["train_job"]
        epochs: List[ModelWeights] = train_job_detailed["epochs"]

        prices = json.loads(train_job.backtest_prices)
        kline_open_time = json.loads(train_job.backtest_kline_open_times)
        predictions = json.loads(epochs[backtestInfo.epoch_nr - 1]["val_predictions"])

        assert len(prices) == len(predictions)

        replacements = {
            "{ENTER_AND_EXIT_CRITERIA_FUNCS}": backtestInfo.enter_trade_cond
            + "\n"
            + backtestInfo.exit_trade_cond,
            "{PREDICTION}": None,
        }

        START_BALANCE = 10000

        backtest_v2 = BacktestV2(START_BALANCE, 0.1, 0.1, replacements)
        for idx in range(len(prices)):
            price = prices[idx]
            prediction = predictions[idx]
            backtest_v2.enter_kline(price, prediction[0], kline_open_time[idx])

        end_balance = backtest_v2.positions.total_positions_value

        backtest_id = BacktestQuery.create_backtest_entry(
            backtestInfo.enter_trade_cond,
            backtestInfo.exit_trade_cond,
            backtest_v2.positions.balance_history,
            epochs[backtestInfo.epoch_nr]["id"],
            train_job.id,
            START_BALANCE,
            end_balance,
        )
        TradeQuery.create_many_trade_entry(backtest_id, backtest_v2.positions.trades)

        backtest_from_db = BacktestQuery.fetch_backtest_by_id(backtest_id)
        return backtest_from_db


class Positions:
    def __init__(self, start_balance, fees, slippage):
        self.cash = start_balance
        self.fees = fees
        self.slippage = slippage
        self.position = 0.0
        self.short_debt = 0.0
        self.total_positions_value = 0.0

        self.enter_trade_price = 0.0
        self.enter_trade_time = 0
        self.enter_trade_prediction = 0.0
        self.enter_trade_balance = 0.0

        self.trades = []
        self.balance_history = []
        self.trade_prices = []
        self.trade_predictions = []

    def close_long(self, price: float, kline_open_time: int):
        self.cash += price * self.position
        self.position = 0.0
        self.cash = self.cash * self.slippage * self.fees
        self.add_trade(price, kline_open_time, Direction.SHORT)

    def reset_trade_track_data(self):
        self.trade_prices = []
        self.trade_predictions = []

    def add_trade(self, price: float, kline_open_time: int, direction):
        net_profit = self.total_positions_value - self.enter_trade_balance
        percent_result = (
            self.total_positions_value / self.enter_trade_balance - 1
        ) * 100

        self.trades.append(
            {
                "open_price": self.enter_trade_price,
                "close_price": price,
                "open_time": self.enter_trade_time,
                "close_time": kline_open_time,
                "direction": direction,
                "net_result": net_profit,
                "percent_result": percent_result,
                "predictions": self.trade_predictions,
                "prices": self.trade_prices,
            }
        )

    def close_short(self, price: float, kline_open_time: int):
        price_to_close = price * self.short_debt
        self.cash -= price_to_close
        self.short_debt = 0.0
        self.cash = self.cash * self.slippage * self.fees

        self.add_trade(price, kline_open_time, Direction.SHORT)

    def init_trade_track_data(
        self, price: float, prediction: float, kline_open_time: int
    ):
        self.enter_trade_time = kline_open_time
        self.enter_trade_price = price
        self.enter_trade_prediction = prediction
        self.enter_trade_balance = self.total_positions_value
        self.trade_prices = [price]
        self.trade_predictions = [prediction]

    def go_long(self, price: float, prediction: float, kline_open_time: int):
        self.cash = self.cash * self.slippage * self.fees
        self.position = self.cash / price
        self.cash = 0.0

        self.init_trade_track_data(price, prediction, kline_open_time)

    def go_short(self, price: float, prediction: float, kline_open_time: int):
        self.cash = self.cash * self.slippage * self.fees
        position_size = self.cash / price
        short_proceedings = position_size * price
        self.short_debt = position_size
        self.cash += short_proceedings

        self.init_trade_track_data(price, prediction, kline_open_time)

    def update_balance(self, price: float, prediction: float, kline_open_time: int):
        portfolio_worth = self.cash
        if self.position > 0.0:
            portfolio_worth += price * self.position
        if self.short_debt > 0.0:
            portfolio_worth -= price * self.short_debt
        self.total_positions_value = portfolio_worth
        self.balance_history.append(
            {
                "portfoli_worth": portfolio_worth,
                "prediction": prediction,
                "kline_open_time": kline_open_time,
                "position": self.position,
                "short_debt": self.short_debt,
                "cash": self.cash,
            }
        )


class BacktestV2:
    def __init__(
        self,
        start_balance: float,
        fees_perc: float,
        slippage_perc: float,
        enter_and_exit_criteria_placeholders: Dict,
    ) -> None:
        self.enter_and_exit_criteria_placeholders = enter_and_exit_criteria_placeholders
        self.positions = Positions(
            start_balance, 1 - (fees_perc / 100), 1 - (slippage_perc / 100)
        )
        self.history: List = []

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

    def update_data(self, price: float, prediction: float, kline_open_time: int):
        if self.positions.position > 0.0 or self.positions.short_debt > 0.0:
            self.positions.trade_predictions.append(prediction)
            self.positions.trade_prices.append(prediction)
        self.positions.update_balance(price, prediction, kline_open_time)

    def tick(
        self,
        price: float,
        prediction: float,
        kline_open_time: int,
        should_long: bool,
        should_short: bool,
    ):
        self.update_data(price, prediction, kline_open_time)

        if self.positions.position > 0 and should_long is False:
            self.positions.close_long(price, kline_open_time)

        if self.positions.short_debt > 0 and should_short is False:
            self.positions.close_short(price, kline_open_time)

        if self.positions.cash > 0 and should_long is True:
            self.positions.go_long(price, prediction, kline_open_time)

        if self.positions.short_debt == 0 and should_short is True:
            self.positions.go_short(price, prediction, kline_open_time)
