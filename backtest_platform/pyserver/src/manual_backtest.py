import logging
import math
from typing import Dict, List
from api_binance import save_historical_klines
from query_backtest_history import BacktestHistoryQuery
from backtest_utils import (
    find_max_drawdown,
    get_backtest_profit_factor_comp,
    get_backtest_trade_details,
    get_cagr,
    turn_short_fee_perc_to_coeff,
)
from code_gen_template import BACKTEST_MANUAL_TEMPLATE
from constants import BINANCE_BACKTEST_PRICE_COL, YEAR_IN_MS, DomEventChannels
from dataset import read_dataset_to_mem
from db import exec_python, get_df_candle_size, ms_to_years
from log import LogExceptionContext, get_logger
from math_utils import calculate_avg_trade_hold_time_ms, calculate_psr, calculate_sr
from model_backtest import Positions
from query_backtest import BacktestQuery
from query_backtest_statistics import BacktestStatisticsQuery
from query_data_transformation import DataTransformationQuery
from query_dataset import DatasetQuery
from query_mass_backtest import MassBacktestQuery
from query_trade import TradeQuery
from request_types import BodyCreateManualBacktest, BodyCreateMassBacktest
from utils import PythonCode, get_binance_dataset_tablename


START_BALANCE = 10000


async def run_rule_based_mass_backtest(
    mass_backtest_id: int,
    body: BodyCreateMassBacktest,
    interval,
    original_backtest: Dict,
):
    with LogExceptionContext():
        logger = get_logger()
        n_symbols = len(body.symbols)
        curr_iter = 1

        original_dataset = DatasetQuery.fetch_dataset_by_id(
            original_backtest["dataset_id"]
        )
        data_transformations = DataTransformationQuery.get_transformations_by_dataset(
            original_dataset.id
        )

        for symbol in body.symbols:
            table_name = get_binance_dataset_tablename(symbol, interval)
            symbol_dataset = DatasetQuery.fetch_dataset_by_name(table_name)

            if symbol_dataset is None or body.fetch_latest_data is True:
                await save_historical_klines(symbol, interval, True)
                symbol_dataset = DatasetQuery.fetch_dataset_by_name(table_name)

            DatasetQuery.update_price_column(table_name, BINANCE_BACKTEST_PRICE_COL)

            for transformation in data_transformations:
                python_program = PythonCode.on_dataset(
                    table_name, transformation.transformation_code
                )
                exec_python(python_program)
                DataTransformationQuery.create_entry(
                    {
                        "transformation_code": transformation.transformation_code,
                        "dataset_id": symbol_dataset.id,
                    }
                )

            backtest_body = {
                "backtest_data_range": [
                    original_backtest["backtest_range_start"],
                    original_backtest["backtest_range_end"],
                ],
                "open_trade_cond": original_backtest["open_trade_cond"],
                "close_trade_cond": original_backtest["close_trade_cond"],
                "is_short_selling_strategy": original_backtest[
                    "is_short_selling_strategy"
                ],
                "use_time_based_close": original_backtest["use_time_based_close"],
                "use_profit_based_close": original_backtest["use_profit_based_close"],
                "use_stop_loss_based_close": original_backtest[
                    "use_stop_loss_based_close"
                ],
                "dataset_id": symbol_dataset.id,
                "trading_fees_perc": original_backtest["trading_fees_perc"],
                "slippage_perc": original_backtest["slippage_perc"],
                "short_fee_hourly": original_backtest["short_fee_hourly"],
                "take_profit_threshold_perc": original_backtest[
                    "take_profit_threshold_perc"
                ],
                "stop_loss_threshold_perc": original_backtest[
                    "stop_loss_threshold_perc"
                ],
                "name": original_backtest["name"],
                "klines_until_close": original_backtest["klines_until_close"],
            }

            backtest = run_manual_backtest(BodyCreateManualBacktest(**backtest_body))
            MassBacktestQuery.add_backtest_id(mass_backtest_id, backtest["id"])

            logger.log(
                f"Finished backtest ({curr_iter}/{n_symbols}) on {symbol}",
                logging.INFO,
                True,
                True,
                DomEventChannels.REFETCH_COMPONENT.value,
            )
            curr_iter += 1


def run_manual_backtest(backtestInfo: BodyCreateManualBacktest):
    with LogExceptionContext():
        dataset = DatasetQuery.fetch_dataset_by_id(backtestInfo.dataset_id)
        dataset_df = read_dataset_to_mem(dataset.dataset_name)

        candles_time_delta = get_df_candle_size(
            dataset_df, dataset.timeseries_column, formatted=False
        )
        candle_interval = get_df_candle_size(
            dataset_df, dataset.timeseries_column, formatted=True
        )

        replacements = {
            "{OPEN_TRADE_FUNC}": backtestInfo.open_trade_cond,
            "{CLOSE_TRADE_FUNC}": backtestInfo.close_trade_cond,
        }

        backtest = ManualBacktest(
            START_BALANCE,
            backtestInfo.trading_fees_perc,
            backtestInfo.slippage_perc,
            backtestInfo.short_fee_hourly,
            replacements,
            backtestInfo.is_short_selling_strategy,
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

        backtest_stats_dict = {
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "start_balance": START_BALANCE,
            "end_balance": end_balance,
            "result_perc": (end_balance / START_BALANCE - 1) * 100,
            "take_profit_threshold_perc": backtestInfo.take_profit_threshold_perc,
            "stop_loss_threshold_perc": backtestInfo.stop_loss_threshold_perc,
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
            "sharpe_ratio": calculate_sr(
                [x["percent_result"] / 100 for x in backtest.positions.trades],
                rf=0,
                periods_per_year=round(
                    YEAR_IN_MS
                    / calculate_avg_trade_hold_time_ms(backtest.positions.trades)
                )
                if len(backtest.positions.trades) != 0
                else 0,
            ),
            "probabilistic_sharpe_ratio": calculate_psr(
                [x["percent_result"] / 100 for x in backtest.positions.trades],
                sr_star=0,
                periods_per_year=round(
                    YEAR_IN_MS
                    / calculate_avg_trade_hold_time_ms(backtest.positions.trades)
                )
                if len(backtest.positions.trades) != 0
                else 0,
            ),
            "share_of_winning_trades_perc": share_of_winning_trades_perc,
            "share_of_losing_trades_perc": share_of_losing_trades_perc,
            "max_drawdown_perc": max_drawdown_perc,
            "cagr": cagr,
            "market_exposure_time": market_exposure_time,
            "risk_adjusted_return": cagr / market_exposure_time
            if market_exposure_time != 0
            else 0,
            "buy_and_hold_cagr": buy_and_hold_cagr,
            "slippage_perc": backtestInfo.slippage_perc,
            "short_fee_hourly": backtestInfo.short_fee_hourly,
            "trading_fees_perc": backtestInfo.trading_fees_perc,
            "trade_count": len(backtest.positions.trades),
        }

        backtest_id = BacktestQuery.create_entry(
            {
                "open_trade_cond": backtestInfo.open_trade_cond,
                "close_trade_cond": backtestInfo.close_trade_cond,
                "dataset_id": dataset.id,
                "dataset_name": dataset.dataset_name,
                "name": backtestInfo.name,
                "klines_until_close": backtestInfo.klines_until_close,
                "backtest_range_start": backtestInfo.backtest_data_range[0],
                "candle_interval": candle_interval,
                "backtest_range_end": backtestInfo.backtest_data_range[1],
                "use_time_based_close": backtestInfo.use_time_based_close,
                "use_profit_based_close": backtestInfo.use_profit_based_close,
                "use_stop_loss_based_close": backtestInfo.use_stop_loss_based_close,
                "is_short_selling_strategy": backtestInfo.is_short_selling_strategy,
            }
        )

        backtest_stats_dict["backtest_id"] = backtest_id
        BacktestStatisticsQuery.create_entry(backtest_stats_dict)

        backtest_from_db = BacktestQuery.fetch_backtest_by_id(backtest_id)
        TradeQuery.create_many(backtest_id, backtest.positions.trades)
        BacktestHistoryQuery.create_many(
            backtest_id, backtest.positions.balance_history
        )

        return backtest_from_db


class ManualBacktest:
    def __init__(
        self,
        start_balance: float,
        fees_perc: float,
        slippage_perc: float,
        short_fee_hourly_perc: float,
        enter_and_exit_criteria_placeholders: Dict,
        is_short_selling_strategy: bool,
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

        self.enter_and_exit_criteria_placeholders = enter_and_exit_criteria_placeholders
        self.positions = Positions(
            start_balance,
            1 - (fees_perc / 100),
            1 - (slippage_perc / 100),
            short_fee_hourly_coeff,
        )
        self.history: List = []
        self.is_short_selling_strategy = is_short_selling_strategy
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
        should_open_trade = (
            results_dict["should_open_trade"] if is_last_row is False else False
        )
        should_close_trade = (
            results_dict["should_close_trade"] if is_last_row is False else True
        )

        kline_open_time = df_row[timeseries_col]
        price = df_row[price_col]

        if (
            self.pos_open_klines == self.max_klines_until_close
            and self.use_time_based_close
        ):
            # auto close positions if time threshold is met
            should_close_trade = True

        if self.use_profit_based_close and self.positions.take_profit_threshold_hit(
            price, self.take_profit_threshold_perc
        ):
            should_close_trade = True

        if self.use_stop_loss_based_close and self.positions.stop_loss_threshold_hit(
            price, self.stop_loss_threshold_perc
        ):
            should_close_trade = True

        if self.positions.is_trading_forced_stop is True:
            should_close_trade = True

        self.tick(price, kline_open_time, should_open_trade, should_close_trade)

    def post_trade_cleanup(self):
        self.pos_open_klines = 0

    def update_data(self, price: float, kline_open_time: int):
        if self.positions.position > 0.0 or self.positions.short_debt > 0.0:
            self.pos_open_klines += 1
            self.positions_held_time += self.candles_time_delta

        self.positions.update_balance(price, 0, kline_open_time)
        self.cumulative_time += self.candles_time_delta

    def tick(
        self,
        price: float,
        kline_open_time: int,
        should_open_trade: bool,
        should_close_trade: bool,
    ):
        self.update_data(price, kline_open_time)

        if self.is_short_selling_strategy:
            if self.positions.short_debt > 0 and should_close_trade is True:
                self.positions.close_short(price, kline_open_time)
                self.post_trade_cleanup()
            if self.positions.short_debt == 0 and should_open_trade is True:
                self.positions.go_short(price, 0, kline_open_time)
        else:
            if self.positions.position > 0 and should_close_trade:
                self.positions.close_long(price, kline_open_time)

            if self.positions.cash > 0 and should_open_trade:
                self.positions.go_long(price, 0, kline_open_time)
                self.post_trade_cleanup()
