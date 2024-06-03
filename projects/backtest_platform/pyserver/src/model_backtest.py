from typing import Dict, List


from log import LogExceptionContext
from query_backtest_history import BacktestHistoryQuery
from query_backtest_statistics import BacktestStatisticsQuery
from query_epoch_prediction import EpochPredictionQuery
from query_trade import TradeQuery
from query_weights import ModelWeights, ModelWeightsQuery
from query_trainjob import TrainJob, TrainJobQuery
from query_backtest import BacktestQuery
from request_types import BodyRunBacktest
from code_gen_template import BACKTEST_MODEL_TEMPLATE


class Direction:
    LONG = "long"
    SHORT = "short"
    CASH = "cash"


START_BALANCE = 10000


def run_model_backtest(train_job_id: int, backtestInfo: BodyRunBacktest):
    with LogExceptionContext():
        train_job_detailed = TrainJobQuery.get_train_job_detailed(train_job_id)
        train_job: TrainJob = train_job_detailed["train_job"]
        modelweights_metadata = ModelWeightsQuery.fetch_metadata_by_epoch(
            train_job_id, backtestInfo.epoch_nr
        )
        epoch_val_preds = EpochPredictionQuery.get_entries_by_weights_id_sorted(
            modelweights_metadata.id
        )

        prices = [item.price for item in train_job_detailed["validation_set_ticks"]]
        kline_open_time = [
            item.kline_open_time for item in train_job_detailed["validation_set_ticks"]
        ]
        predictions = [item.prediction for item in epoch_val_preds]
        assert len(prices) == len(predictions)

        replacements = {
            "{ENTER_AND_EXIT_CRITERIA_FUNCS}": backtestInfo.enter_trade_cond
            + "\n"
            + backtestInfo.exit_trade_cond,
            "{PREDICTION}": None,
        }

        backtest_v2 = BacktestV2(START_BALANCE, 0.1, 0.1, replacements)
        for idx in range(len(prices)):
            price = prices[idx]
            prediction = predictions[idx]
            backtest_v2.enter_kline(price, prediction, kline_open_time[idx])

        end_balance = backtest_v2.positions.total_positions_value

        backtest_id = BacktestQuery.create_entry(
            {
                "open_trade_cond": backtestInfo.enter_trade_cond,
                "close_trade_cond": backtestInfo.exit_trade_cond,
                "model_weights_id": modelweights_metadata.id,
                "train_job_id": train_job.id,
            }
        )

        backtest_stats_dict = {
            "start_balance": START_BALANCE,
            "end_balance": end_balance,
            "backtest_id": backtest_id,
        }

        TradeQuery.create_many(backtest_id, backtest_v2.positions.trades)
        BacktestStatisticsQuery.create_entry(backtest_stats_dict)
        BacktestHistoryQuery.create_many(
            backtest_id, backtest_v2.positions.balance_history
        )

        backtest_from_db = BacktestQuery.fetch_backtest_by_id(backtest_id)
        return backtest_from_db


class Positions:
    def __init__(self, start_balance, fees, slippage, short_fee_coeff=1):
        self.start_balance = start_balance
        self.cash = start_balance
        self.fees = fees
        self.slippage = slippage
        self.position = 0.0
        self.short_debt = 0.0
        self.short_fee_coeff = short_fee_coeff
        self.total_positions_value = 0.0
        self.buy_and_hold_position = None
        self.is_trading_forced_stop = False

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
        self.reset_trade_track_data()

    def reset_trade_track_data(self):
        self.trade_prices = []
        self.trade_predictions = []

    def add_trade(self, price: float, kline_open_time: int, direction):
        net_profit = self.total_positions_value - self.enter_trade_balance
        percent_result = (
            (self.total_positions_value / self.enter_trade_balance - 1) * 100
            if self.enter_trade_balance != 0
            else 0
        )

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
        self.add_trade(price, kline_open_time, Direction.LONG)
        self.reset_trade_track_data()

    def init_trade_track_data(
        self, price: float, prediction: float, kline_open_time: int
    ):
        self.enter_trade_time = kline_open_time
        self.enter_trade_price = price
        self.enter_trade_prediction = prediction
        self.enter_trade_balance = self.total_positions_value
        self.trade_prices = [price]
        self.trade_predictions = [prediction]

    def take_profit_threshold_hit(
        self, price: float, take_profit_threshold_perc: float
    ):
        if self.position > 0:
            return price / self.enter_trade_price > (
                1 + (take_profit_threshold_perc / 100)
            )

        if self.short_debt > 0:
            return price / self.enter_trade_price < (
                1 - (take_profit_threshold_perc / 100)
            )

        return False

    def stop_loss_threshold_hit(self, price: float, stop_loss_threshold_perc: float):
        if self.position == 0 and self.short_debt == 0:
            return False

        if len(self.trade_prices) == 0:
            return False

        is_short_selling_strat = False

        if self.short_debt > 0.0:
            is_short_selling_strat = True

        prices = self.trade_prices.copy()
        prices.append(price)
        position_ath_price = prices[0]

        for item in prices:
            if self.position > 0:
                if item / position_ath_price < (1 - stop_loss_threshold_perc / 100):
                    return True
            else:
                if item / position_ath_price > (1 + (stop_loss_threshold_perc / 100)):
                    return True
            position_ath_price = (
                max(item, position_ath_price)
                if is_short_selling_strat is False
                else min(item, position_ath_price)
            )
        return False

    def go_long(self, price: float, prediction: float, kline_open_time: int):
        if not self.is_trading_forced_stop:
            self.cash = self.cash * self.slippage * self.fees
            self.position = self.cash / price
            self.cash = 0.0

            self.init_trade_track_data(price, prediction, kline_open_time)

    def go_short(self, price: float, prediction: float, kline_open_time: int):
        if not self.is_trading_forced_stop:
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
            self.trade_prices.append(price)
        if self.short_debt > 0.0:
            self.short_debt *= self.short_fee_coeff
            portfolio_worth -= price * self.short_debt
            self.trade_prices.append(price)

        if portfolio_worth <= START_BALANCE * 0.2:
            self.is_trading_forced_stop = True

        self.total_positions_value = portfolio_worth

        if self.buy_and_hold_position is None:
            self.buy_and_hold_position = self.start_balance / price

        self.balance_history.append(
            {
                "portfolio_worth": portfolio_worth,
                "buy_and_hold_worth": self.buy_and_hold_position * price,
                "prediction": prediction,
                "kline_open_time": kline_open_time,
                "position": self.position,
                "short_debt": self.short_debt,
                "cash": self.cash,
                "price": price,
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
        code = BACKTEST_MODEL_TEMPLATE
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
