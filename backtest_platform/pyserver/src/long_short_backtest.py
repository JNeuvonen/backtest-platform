from typing import Dict, List, Set
from backtest_utils import calc_long_short_profit_factor, turn_short_fee_perc_to_coeff
from code_gen_template import (
    BACKTEST_LONG_SHORT_BUYS_AND_SELLS,
    BACKTEST_LONG_SHORT_CLOSE_TEMPLATE,
)
from constants import BINANCE_BACKTEST_PRICE_COL, AppConstants
from dataset import (
    get_row_count,
    read_all_cols_matching_kline_open_times,
    read_columns_to_mem,
)
from db import exec_python, get_df_candle_size
from log import LogExceptionContext
from math_utils import safe_divide
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


def get_benchmark_initial_state(datasets: List):
    for item in datasets:
        dataset = DatasetQuery.fetch_dataset_by_id(item)


def get_datasets_kline_state(
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

    id_to_df_map = {}

    with LogExceptionContext():
        for item in backtest_info.datasets:
            dataset_name = dataset_id_to_name_map[str(item)]
            timeseries_col = dataset_id_to_ts_col[str(item)]
            df = read_all_cols_matching_kline_open_times(
                dataset_name, timeseries_col, [kline_open_time]
            )
            id_to_df_map[str(item)] = df

            if df.empty is True:
                continue

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

    return {
        "sell_candidates": sell_candidates,
        "buy_candidates": buy_candidates,
        "id_to_row_map": id_to_df_map,
    }


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

        benchmark_initial_state = get_benchmark_initial_state(backtest_info.datasets)

        long_short_backtest = LongShortOnUniverseBacktest(
            backtest_info,
            candles_time_delta,
            dataset_id_to_name_map,
            dataset_id_to_timeseries_col_map,
        )

        if kline_open_times is None:
            raise Exception("Kline_open_times df was none.")

        for _, row in kline_open_times.iterrows():
            kline_state = get_datasets_kline_state(
                row[timeseries_col],
                backtest_info,
                dataset_id_to_name_map,
                dataset_id_to_timeseries_col_map,
            )

            kline_open_time = row[timeseries_col]
            long_short_backtest.process_bar(
                kline_open_time=kline_open_time, kline_state=kline_state
            )

        profit_factor_dict = calc_long_short_profit_factor(
            long_short_backtest.completed_trades
        )


class BacktestRules:
    def __init__(
        self, backtest_details: BodyCreateLongShortBacktest, candles_time_delta
    ):
        self.buy_cond = backtest_details.buy_cond
        self.sell_cond = backtest_details.sell_cond
        self.exit_cond = backtest_details.exit_cond

        self.use_profit_based_close = backtest_details.use_profit_based_close
        self.use_stop_loss_based_close = backtest_details.use_stop_loss_based_close
        self.use_time_based_close = backtest_details.use_time_based_close

        self.max_klines_until_close = backtest_details.klines_until_close
        self.take_profit_threshold_perc = backtest_details.take_profit_threshold_perc
        self.stop_loss_threshold_perc = backtest_details.stop_loss_threshold_perc

        self.trading_fees = (backtest_details.trading_fees_perc) / 100
        self.short_fee_coeff = turn_short_fee_perc_to_coeff(
            backtest_details.short_fee_hourly, candles_time_delta
        )

        self.max_simultaneous_positions = backtest_details.max_simultaneous_positions
        self.max_leverage_ratio = backtest_details.max_leverage_ratio


class BacktestStats:
    def __init__(self, candles_time_delta: int):
        self.cumulative_time = 0
        self.position_held_time = 0.0
        self.candles_time_delta = candles_time_delta


class PositionManager:
    def __init__(self, usdt_start_balance: float):
        self.usdt_balance = usdt_start_balance
        self.net_value = usdt_start_balance
        self.usdt_debt = 0

        self.open_positions_total_debt = 0
        self.open_positions_total_long = 0
        self.debt_to_net_value_ratio = 0.0

        self.position_history: List = []

    def close_long(self, proceedings: float):
        self.usdt_balance += proceedings

    def close_short(self, proceedings: float):
        self.usdt_balance -= proceedings

    def update(self, positions_debt, positions_long, kline_open_time):
        self.net_value = self.usdt_balance + positions_long - positions_debt
        self.open_positions_total_long = positions_long
        self.open_positions_total_debt = positions_debt
        self.debt_to_net_value_ratio = positions_debt / self.net_value

        tick = {
            "total_debt": positions_debt,
            "total_long": positions_long,
            "kline_open_time": kline_open_time,
            "debt_to_net_value_ratio": self.debt_to_net_value_ratio,
            "net_value": self.net_value,
            "benchmark_price": 0,
        }
        self.position_history.append(tick)


class PairTrade:
    def __init__(
        self,
        buy_id: int,
        buy_amount_base: float,
        buy_amount_quote: float,
        sell_id: int,
        sell_proceedings_quote: float,
        debt_amount_base: float,
        trade_open_time,
        sell_price: float,
        buy_price: float,
        on_trade_open_acc_net_value: float,
    ):
        self.buy_id = buy_id
        self.sell_id = sell_id
        self.buy_amount_base = buy_amount_base
        self.buy_amount_quote = buy_amount_quote
        self.sell_proceedings_quote = sell_proceedings_quote
        self.debt_amount_base = debt_amount_base
        self.trade_open_time = trade_open_time
        self.on_open_sell_price = sell_price
        self.on_open_buy_price = buy_price
        self.on_trade_open_acc_net_value = on_trade_open_acc_net_value
        self.balance_history: List[float] = []


class CompletedTrade:
    def __init__(
        self,
        buy_id: int,
        sell_id: int,
        long_open_price: float,
        long_close_price: float,
        short_open_price: float,
        short_close_price: float,
        on_open_long_side_amount_quote: float,
        on_close_long_side_amount_quote: float,
        on_open_short_side_amount_quote: float,
        on_close_short_side_amount_quote: float,
        on_trade_open_acc_net_value: float,
        balance_history: List[float],
    ):
        self.buy_id = buy_id
        self.sell_id = sell_id

        self.long_open_price = long_open_price
        self.long_close_price = long_close_price

        self.short_open_price = short_open_price
        self.short_close_price = short_close_price

        self.balance_history = balance_history

        self.long_side_gross_result = (
            on_close_long_side_amount_quote - on_open_long_side_amount_quote
        )
        self.long_side_perc_result = (
            safe_divide(
                on_close_long_side_amount_quote,
                on_open_long_side_amount_quote,
                fallback=0,
            )
            * 100
        )
        self.short_side_gross_result = (
            on_open_short_side_amount_quote - on_close_short_side_amount_quote
        )
        self.short_side_perc_result = (
            safe_divide(
                (on_open_short_side_amount_quote - on_close_short_side_amount_quote),
                on_open_short_side_amount_quote,
                fallback=0,
            )
            * 100
        )
        self.trade_gross_result = (
            self.long_side_gross_result + self.short_side_gross_result
        )

        self.perc_result = (
            safe_divide(self.trade_gross_result, on_trade_open_acc_net_value, 0) * 100
        )


class LongShortOnUniverseBacktest:
    def __init__(
        self,
        backtest_details: BodyCreateLongShortBacktest,
        candles_time_delta: int,
        dataset_id_to_name_map: Dict,
        dataset_id_to_timeseries_col_map: Dict,
    ):
        self.rules = BacktestRules(backtest_details, candles_time_delta)
        self.stats = BacktestStats(candles_time_delta)
        self.positions = PositionManager(START_BALANCE)
        self.dataset_id_to_name_map = dataset_id_to_name_map
        self.dataset_id_to_timeseries_col_map = dataset_id_to_timeseries_col_map

        self.history: List = []
        self.active_pairs: List[PairTrade] = []
        self.completed_trades: List = []

    def form_trading_pairs(self, buy_and_sell_candidates: Dict):
        valid_buys = list(buy_and_sell_candidates["buy_candidates"])
        valid_sells = list(buy_and_sell_candidates["sell_candidates"])
        n_valid_sells = len(valid_sells)

        ret = []

        for i in range(len(valid_buys)):
            if i < n_valid_sells:
                valid_buy = valid_buys[i]
                valid_sell = valid_sells[i]
                ret.append({"sell": valid_sell, "buy": valid_buy})
            else:
                break

        return ret

    def get_available_new_pos_size(self):
        current_debt_ratio = self.positions.debt_to_net_value_ratio
        max_total_allocation = self.rules.max_leverage_ratio
        max_invidual_allocation = (
            max_total_allocation / self.rules.max_simultaneous_positions
        )

        if max_total_allocation - current_debt_ratio < 0.0:
            return 0

        new_allocation_size = min(
            max_invidual_allocation, max_total_allocation - current_debt_ratio
        )
        return new_allocation_size * self.positions.net_value

    def remove_pairs_already_in_trade(self, available_pairs):
        active_trade_ids = {trade.buy_id for trade in self.active_pairs} | {
            trade.sell_id for trade in self.active_pairs
        }

        filtered_pairs = [
            item
            for item in available_pairs
            if item["buy"] not in active_trade_ids
            and item["sell"] not in active_trade_ids
        ]

        return filtered_pairs

    def get_exit_pair_trade_decision(self, kline_state: Dict, pair: PairTrade):
        code = BACKTEST_LONG_SHORT_CLOSE_TEMPLATE
        replacements = {"{EXIT_PAIR_TRADE_FUNC}": self.rules.exit_cond}

        for key, value in replacements.items():
            code = code.replace(key, str(value))

        results_dict = {
            "buy_df": kline_state["id_to_row_map"][str(pair.buy_id)].iloc[0],
            "sell_df": kline_state["id_to_row_map"][str(pair.sell_id)].iloc[0],
        }
        exec(code, globals(), results_dict)

        should_close_trade = results_dict["should_close_trade"]
        return should_close_trade

    def get_close_long_proceedings(self, buy_df, pair: PairTrade):
        close_long_amount = (
            buy_df.iloc[0][BINANCE_BACKTEST_PRICE_COL]
            * pair.buy_amount_base
            * self.trading_fees_coeff_reduce_amount()
        )
        return close_long_amount

    def trading_fees_coeff_reduce_amount(self):
        return 1 - self.rules.trading_fees

    def trading_fees_coeff_increase_amount(self):
        return self.rules.trading_fees + 1

    def get_close_short_amount(self, sell_df, pair: PairTrade):
        close_short_amount = (
            sell_df.iloc[0][BINANCE_BACKTEST_PRICE_COL]
            * pair.debt_amount_base
            * self.trading_fees_coeff_increase_amount()
        )
        return close_short_amount

    def get_trade_enter_long_amount(self, usdt_size: float, price: float):
        amount = usdt_size / price
        return amount * self.trading_fees_coeff_reduce_amount()

    def get_bar_curr_price(self, df):
        price = df.iloc[0][BINANCE_BACKTEST_PRICE_COL]
        return price

    def close_pair_trade(self, kline_state: Dict, pair: PairTrade):
        buy_df = kline_state["id_to_row_map"][str(pair.buy_id)]
        sell_df = kline_state["id_to_row_map"][str(pair.sell_id)]

        long_close_price = self.get_bar_curr_price(buy_df)
        short_close_price = self.get_bar_curr_price(sell_df)

        sell_long_proceedings = self.get_close_long_proceedings(buy_df, pair)
        required_for_close_short = self.get_close_short_amount(sell_df, pair)

        completed_trade = CompletedTrade(
            buy_id=pair.buy_id,
            sell_id=pair.sell_id,
            long_open_price=pair.on_open_buy_price,
            long_close_price=long_close_price,
            short_open_price=pair.on_open_sell_price,
            short_close_price=short_close_price,
            balance_history=pair.balance_history,
            on_open_long_side_amount_quote=pair.buy_amount_quote,
            on_open_short_side_amount_quote=pair.sell_proceedings_quote,
            on_close_long_side_amount_quote=sell_long_proceedings,
            on_close_short_side_amount_quote=required_for_close_short,
            on_trade_open_acc_net_value=pair.on_trade_open_acc_net_value,
        )

        self.positions.close_long(sell_long_proceedings)
        self.positions.close_short(required_for_close_short)
        self.completed_trades.append(completed_trade)

    def check_for_pair_trade_close(self, kline_state):
        active_pairs = []
        for item in self.active_pairs:
            should_exit = self.get_exit_pair_trade_decision(kline_state, item)
            if should_exit is True:
                self.close_pair_trade(kline_state, item)
            else:
                active_pairs.append(item)
        self.active_pairs = active_pairs

    def enter_pair_trade(self, kline_open_time, kline_state, pair):
        buy_id = pair["buy"]
        sell_id = pair["sell"]

        buy_df = kline_state["id_to_row_map"][str(buy_id)]
        sell_df = kline_state["id_to_row_map"][str(sell_id)]

        buy_price = self.get_bar_curr_price(buy_df)
        sell_price = self.get_bar_curr_price(sell_df)

        usdt_size = self.get_available_new_pos_size()

        debt_amount_base = usdt_size / sell_price
        sell_proceedings_quote = (
            debt_amount_base * sell_price * self.trading_fees_coeff_reduce_amount()
        )
        buy_amount_base = self.get_trade_enter_long_amount(
            sell_proceedings_quote, buy_price
        )

        new_pair_trade = PairTrade(
            buy_id=buy_id,
            sell_id=sell_id,
            debt_amount_base=debt_amount_base,
            buy_amount_base=buy_amount_base,
            buy_amount_quote=buy_amount_base * buy_price,
            sell_proceedings_quote=sell_proceedings_quote,
            trade_open_time=kline_open_time,
            sell_price=sell_price,
            buy_price=buy_price,
            on_trade_open_acc_net_value=self.positions.net_value,
        )
        self.active_pairs.append(new_pair_trade)

    def enter_trades(self, kline_open_time, kline_state, enter_trade_pairs):
        for pair in enter_trade_pairs:
            self.enter_pair_trade(kline_open_time, kline_state, pair)

    def update_pos_held_time(self):
        capital_usage_coeff = safe_divide(
            len(self.active_pairs), self.rules.max_simultaneous_positions, 0
        )
        bar_pos_held_time = capital_usage_coeff * self.stats.candles_time_delta

        self.stats.position_held_time += bar_pos_held_time
        self.stats.cumulative_time += self.stats.candles_time_delta

    def update_acc_state(self, kline_open_time: int, kline_state: Dict):
        total_debt = 0.0
        total_longs = 0.0

        for item in self.active_pairs:
            item.debt_amount_base *= self.rules.short_fee_coeff

        for item in self.active_pairs:
            buy_df = kline_state["id_to_row_map"][str(item.buy_id)]
            sell_df = kline_state["id_to_row_map"][str(item.sell_id)]

            buy_side_price = self.get_bar_curr_price(buy_df)
            sell_side_price = self.get_bar_curr_price(sell_df)

            total_longs += buy_side_price * item.buy_amount_base
            total_debt += sell_side_price * item.debt_amount_base

        self.positions.update(
            positions_debt=total_debt,
            positions_long=total_longs,
            kline_open_time=kline_open_time,
        )
        self.update_pos_held_time()

    def process_bar(self, kline_open_time: int, kline_state: Dict):
        self.update_acc_state(kline_open_time, kline_state)
        self.check_for_pair_trade_close(kline_state)

        pairs_available = self.remove_pairs_already_in_trade(
            self.form_trading_pairs(kline_state)
        )
        self.enter_trades(kline_open_time, kline_state, pairs_available)
