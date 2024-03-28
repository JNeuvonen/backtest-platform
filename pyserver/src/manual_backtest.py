import json
import math
from typing import Dict, List
from backtest_utils import (
    find_max_drawdown,
    get_backtest_profit_factor_comp,
    get_backtest_trade_details,
    get_cagr,
    turn_short_fee_perc_to_coeff,
)
from code_gen_template import BACKTEST_MANUAL_TEMPLATE
from dataset import read_dataset_to_mem
from db import get_df_candle_size, ms_to_years
from log import LogExceptionContext
from model_backtest import Positions
from query_backtest import BacktestQuery
from query_dataset import DatasetQuery
from query_trade import TradeQuery
from request_types import BodyCreateManualBacktest


START_BALANCE = 10000


def run_manual_backtest(backtestInfo: BodyCreateManualBacktest):
    with LogExceptionContext():
        dataset = DatasetQuery.fetch_dataset_by_id(backtestInfo.dataset_id)
        dataset_df = read_dataset_to_mem(dataset.dataset_name)

        candles_time_delta = get_df_candle_size(
            dataset_df, dataset.timeseries_column, formatted=False
        )

        replacements = {
            "{OPEN_LONG_TRADE_FUNC}": backtestInfo.open_long_trade_cond,
            "{OPEN_SHORT_TRADE_FUNC}": backtestInfo.open_short_trade_cond,
            "{CLOSE_LONG_TRADE_FUNC}": backtestInfo.close_long_trade_cond,
            "{CLOSE_SHORT_TRADE_FUNC}": backtestInfo.close_short_trade_cond,
        }

        backtest = ManualBacktest(
            START_BALANCE,
            backtestInfo.trading_fees_perc,
            backtestInfo.slippage_perc,
            backtestInfo.short_fee_hourly,
            replacements,
            backtestInfo.use_short_selling,
            backtestInfo.use_time_based_close,
            backtestInfo.use_profit_based_close,
            backtestInfo.use_stop_loss_based_close,
            backtestInfo.take_profit_threshold_perc,
            backtestInfo.stop_loss_threshold_perc,
            backtestInfo.klines_until_close if backtestInfo.klines_until_close else -1,
            candles_time_delta,
        )

        backtest_data_range_start, backtest_data_range_end = (
            backtestInfo.backtest_data_range + [None, None]
        )[:2]

        assert (
            backtest_data_range_start is not None
        ), "Backtest data range start is missing"
        assert backtest_data_range_end is not None, "Backtest data range end is missing"
        assert (
            dataset.timeseries_column is not None
        ), "Timeseries column has not been set"
        assert dataset.price_column is not None, "Price column has not been set"

        asset_starting_price = None
        asset_closing_price = None

        backtest_data_range_start = math.floor(
            len(dataset_df) * (backtest_data_range_start / 100)
        )
        backtest_data_range_end = math.floor(
            len(dataset_df) * (backtest_data_range_end / 100)
        )
        idx = 0

        for i, row in dataset_df.iterrows():
            is_last_row = i == len(dataset_df) - 1 or idx == backtest_data_range_end

            if i == 0:
                asset_starting_price = row[dataset.price_column]
            if is_last_row:
                asset_closing_price = row[dataset.price_column]

            idx += 1
            if idx <= backtest_data_range_start or idx > backtest_data_range_end:
                continue

            backtest.process_df_row(
                row, dataset.price_column, dataset.timeseries_column, is_last_row
            )

        end_balance = backtest.positions.total_positions_value

        profit_factor, gross_profit, gross_loss = get_backtest_profit_factor_comp(
            backtest.positions.trades
        )
        (
            share_of_winning_trades_perc,
            share_of_losing_trades_perc,
            best_trade_result_perc,
            worst_trade_result_perc,
        ) = get_backtest_trade_details(backtest.positions.trades)

        max_drawdown_perc = find_max_drawdown(backtest.positions.balance_history)
        cagr = get_cagr(
            end_balance, START_BALANCE, ms_to_years(backtest.cumulative_time)
        )
        market_exposure_time = backtest.positions_held_time / backtest.cumulative_time
        buy_and_hold_cagr = get_cagr(
            asset_closing_price,
            asset_starting_price,
            ms_to_years(backtest.cumulative_time),
        )

        backtest_id = BacktestQuery.create_entry(
            {
                "open_long_trade_cond": backtestInfo.open_long_trade_cond,
                "open_short_trade_cond": backtestInfo.open_short_trade_cond,
                "close_long_trade_cond": backtestInfo.close_long_trade_cond,
                "close_short_trade_cond": backtestInfo.close_short_trade_cond,
                "data": json.dumps(backtest.positions.balance_history),
                "dataset_id": dataset.id,
                "start_balance": START_BALANCE,
                "end_balance": end_balance,
                "profit_factor": profit_factor,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "trade_count": len(backtest.positions.trades),
                "name": backtestInfo.name,
                "klines_until_close": backtestInfo.klines_until_close,
                "result_perc": (end_balance / START_BALANCE - 1) * 100,
                "share_of_winning_trades_perc": share_of_winning_trades_perc,
                "share_of_losing_trades_perc": share_of_losing_trades_perc,
                "best_trade_result_perc": best_trade_result_perc,
                "worst_trade_result_perc": worst_trade_result_perc,
                "buy_and_hold_result_net": (
                    (asset_closing_price / asset_starting_price * START_BALANCE)
                    - START_BALANCE
                )
                if asset_starting_price is not None and asset_closing_price is not None
                else None,
                "buy_and_hold_result_perc": (
                    (asset_closing_price / asset_starting_price - 1) * 100
                )
                if asset_starting_price is not None and asset_closing_price is not None
                else None,
                "max_drawdown_perc": max_drawdown_perc,
                "cagr": cagr,
                "market_exposure_time": market_exposure_time,
                "risk_adjusted_return": cagr / market_exposure_time,
                "buy_and_hold_cagr": buy_and_hold_cagr,
                "use_time_based_close": backtestInfo.use_time_based_close,
                "use_profit_based_close": backtestInfo.use_profit_based_close,
                "use_stop_loss_based_close": backtestInfo.use_stop_loss_based_close,
                "stop_loss_threshold_perc": backtestInfo.stop_loss_threshold_perc,
                "take_profit_threshold_perc": backtestInfo.take_profit_threshold_perc,
                "use_short_selling": backtestInfo.use_short_selling,
            }
        )

        backtest_from_db = BacktestQuery.fetch_backtest_by_id(backtest_id)
        TradeQuery.create_many_trade_entry(backtest_id, backtest.positions.trades)

        return backtest_from_db


class ManualBacktest:
    def __init__(
        self,
        start_balance: float,
        fees_perc: float,
        slippage_perc: float,
        short_fee_hourly_perc: float,
        enter_and_exit_criteria_placeholders: Dict,
        use_short_selling: bool,
        use_time_based_close: bool,
        use_profit_based_close: bool,
        use_stop_loss_based_close: bool,
        take_profit_threshold_perc,
        stop_loss_threshold_perc,
        max_klines_until_close: int,
        candles_time_delta,
    ) -> None:
        short_fee_hourly_coeff = turn_short_fee_perc_to_coeff(
            short_fee_hourly_perc, candles_time_delta
        )
        print(short_fee_hourly_coeff)

        self.enter_and_exit_criteria_placeholders = enter_and_exit_criteria_placeholders
        self.positions = Positions(
            start_balance,
            1 - (fees_perc / 100),
            1 - (slippage_perc / 100),
            short_fee_hourly_coeff,
        )
        self.history: List = []
        self.use_short_selling = use_short_selling
        self.use_time_based_close = use_time_based_close
        self.max_klines_until_close = max_klines_until_close
        self.pos_open_klines = 0

        self.candles_time_delta = candles_time_delta
        self.cumulative_time = 0.0
        self.positions_held_time = 0.0

        self.use_profit_based_close = use_profit_based_close
        self.use_stop_loss_based_close = use_stop_loss_based_close
        self.take_profit_threshold_perc = take_profit_threshold_perc
        self.stop_loss_threshold_perc = stop_loss_threshold_perc

    def process_df_row(self, df_row, price_col, timeseries_col, is_last_row):
        code = BACKTEST_MANUAL_TEMPLATE
        for key, value in self.enter_and_exit_criteria_placeholders.items():
            code = code.replace(key, str(value))

        results_dict = {"df_row": df_row}
        exec(code, globals(), results_dict)

        ## Force close trades on last row to make accounting easier
        should_open_long = (
            results_dict["should_open_long"] if is_last_row is False else False
        )
        should_open_short = (
            results_dict["should_open_short"] if is_last_row is False else False
        )
        should_close_long = (
            results_dict["should_close_long"] if is_last_row is False else True
        )
        should_close_short = (
            results_dict["should_close_short"] if is_last_row is False else True
        )

        kline_open_time = df_row[timeseries_col]
        price = df_row[price_col]

        if (
            self.pos_open_klines == self.max_klines_until_close
            and self.use_time_based_close
        ):
            # auto close positions if time threshold is met
            should_close_short = True
            should_close_long = True

        if self.use_profit_based_close and self.positions.take_profit_threshold_hit(
            price, self.take_profit_threshold_perc
        ):
            should_close_long = True
            should_close_short = True

        if self.use_stop_loss_based_close and self.positions.stop_loss_threshold_hit(
            price, self.stop_loss_threshold_perc
        ):
            should_close_long = True
            should_close_short = True

        if self.positions.is_trading_forced_stop is True:
            should_close_long = True
            should_close_short = True

        self.tick(
            price,
            kline_open_time,
            should_open_long,
            should_open_short,
            should_close_long,
            should_close_short,
        )

    def close_trade_cleanup(self):
        self.pos_open_klines = 0

    def update_data(self, price: float, kline_open_time: int):
        if self.positions.position > 0.0 or self.positions.short_debt > 0.0:
            self.positions_held_time += self.candles_time_delta

        self.positions.update_balance(price, 0, kline_open_time)
        self.pos_open_klines += 1
        self.cumulative_time += self.candles_time_delta

    def tick(
        self,
        price: float,
        kline_open_time: int,
        should_long: bool,
        should_short: bool,
        should_close_long: bool,
        should_close_short: bool,
    ):
        self.update_data(price, kline_open_time)

        if self.positions.position > 0 and should_close_long:
            self.positions.close_long(price, kline_open_time)

        if (
            self.positions.short_debt > 0
            and should_close_short is True
            and self.use_short_selling is True
        ):
            self.positions.close_short(price, kline_open_time)
            self.close_trade_cleanup()

        if self.positions.cash > 0 and should_long:
            self.positions.go_long(price, 0, kline_open_time)
            self.close_trade_cleanup()

        if (
            self.positions.short_debt == 0
            and should_short is True
            and self.use_short_selling is True
        ):
            self.positions.go_short(price, 0, kline_open_time)
