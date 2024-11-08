from typing import List
from common_python.log import LogLevel, create_log
import pandas as pd
from binance import Client
from prediction_server.log import LogExceptionContext
from common_python.pred_serv_orm import engine
from prediction_server.dataset_utils import read_dataset_to_mem, read_latest_row
from prediction_server.code_gen_templates import CodeTemplates
from common_python.pred_serv_models.data_transformation import (
    DataTransformationQuery,
)
from common_python.pred_serv_models.strategy import StrategyQuery
from common_python.pred_serv_models.longshortticker import LongShortTickerQuery
from common_python.pred_serv_models.longshortpair import LongShortPairQuery
from prediction_server.rule_based_loop_utils import LocalDataset, RuleBasedLoopManager

from prediction_server.utils import (
    NUM_REQ_KLINES_BUFFER,
    calculate_timestamp_for_kline_fetch,
    gen_data_transformations_code,
    get_current_timestamp_ms,
    replace_placeholders_on_code_templ,
)


client = Client()


DATABASES_DIR = "databases"


BINANCE_DATA_COLS = [
    "kline_open_time",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "kline_close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

BINANCE_KLINES_MAX_LIMIT = 1000


def fetch_binance_klines(symbol, interval, start_str):
    klines = []

    start_time = start_str
    requests = 0

    while True:
        new_klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_time,
            limit=BINANCE_KLINES_MAX_LIMIT,
        )

        requests += 1

        if not new_klines:
            break

        klines += new_klines
        start_time = int(new_klines[-1][0]) + 1

    klines = klines[:-1] if klines else klines

    df = pd.DataFrame(klines, columns=BINANCE_DATA_COLS)

    convert_orig_cols_to_numeric(df)
    return df


def get_last_kline_open_time(klines_df):
    if not klines_df.empty:
        last_kline_open_time_ms = klines_df.iloc[-1]["kline_open_time"]
        return last_kline_open_time_ms / 1000
    return None


def transform_and_predict(strategy, df, local_dataset: LocalDataset):
    if strategy.num_req_klines > df.shape[0]:
        return {
            "should_enter_trade": False,
            "should_close_trade": False,
            "is_on_pred_serv_err": False,
        }

    if strategy.stop_processing_new_candles is True:
        return {
            "should_enter_trade": strategy.should_enter_trade,
            "should_close_trade": strategy.should_exit_trade,
            "is_on_pred_serv_err": False,
        }

    data_transformations = local_dataset.transformations

    results_dict = {
        "transformed_data": None,
        "fetched_data": df,
        "should_enter_trade": None,
        "should_exit_trade": None,
    }

    for item in data_transformations:
        data_transformations_helper = {
            "{DATA_TRANSFORMATIONS_FUNC}": gen_data_transformations_code([item])
        }
        exec(
            replace_placeholders_on_code_templ(
                CodeTemplates.DATA_TRANSFORMATIONS, data_transformations_helper
            ),
            globals(),
            results_dict,
        )

    trade_decision_helper = {
        "{ENTER_TRADE_FUNC}": strategy.enter_trade_code,
        "{EXIT_TRADE_FUNC}": strategy.exit_trade_code,
    }

    exec(
        replace_placeholders_on_code_templ(
            CodeTemplates.GEN_TRADE_DECISIONS, trade_decision_helper
        ),
        globals(),
        results_dict,
    )

    return {
        "should_enter_trade": results_dict["should_enter_trade"],
        "should_close_trade": results_dict["should_exit_trade"],
        "is_on_pred_serv_err": False,
    }


def convert_orig_cols_to_numeric(df):
    if not df.empty:
        for col in BINANCE_DATA_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def long_short_transform_and_predict(longshort_strategy, klines, transformations):
    results_dict = {
        "fetched_data": klines,
        "transformed_data": None,
        "is_valid_buy": False,
        "is_valid_sell": False,
    }

    for item in transformations:
        data_transformations_helper = {
            "{DATA_TRANSFORMATIONS_FUNC}": gen_data_transformations_code([item])
        }
        exec(
            replace_placeholders_on_code_templ(
                CodeTemplates.DATA_TRANSFORMATIONS, data_transformations_helper
            ),
            globals(),
            results_dict,
        )

    trade_decision_helper = {
        "{IS_VALID_BUY_FUNC}": longshort_strategy.buy_cond,
        "{IS_VALID_SELL_FUNC}": longshort_strategy.sell_cond,
    }

    exec(
        replace_placeholders_on_code_templ(
            CodeTemplates.LONG_SHORT_BUY_AND_SELL, trade_decision_helper
        ),
        globals(),
        results_dict,
    )

    return {
        "is_on_pred_serv_err": False,
        "is_valid_buy": results_dict["is_valid_buy"],
        "is_valid_sell": results_dict["is_valid_sell"],
    }, results_dict["transformed_data"]


def long_short_process_ticker(longshort_strategy, transformations, ticker):
    if ticker.last_kline_open_time_sec is None:
        klines = fetch_binance_klines(
            ticker.symbol,
            longshort_strategy.candle_interval,
            calculate_timestamp_for_kline_fetch(
                longshort_strategy.num_req_klines, longshort_strategy.kline_size_ms
            ),
        )

        last_kline_open_time_sec = get_last_kline_open_time(klines)

        decisions, transformed_data = long_short_transform_and_predict(
            longshort_strategy, klines, transformations
        )

        transformed_data.to_sql(
            ticker.dataset_name, con=engine, if_exists="replace", index=False
        )

        LongShortTickerQuery.update(
            ticker.id,
            {"last_kline_open_time_sec": last_kline_open_time_sec, **decisions},
        )
        return decisions
    else:
        last_kline_open_time_ms = (ticker.last_kline_open_time_sec * 1000) + 1
        current_time_ms = get_current_timestamp_ms()

        if (
            current_time_ms
            >= last_kline_open_time_ms + longshort_strategy.kline_size_ms * 2
        ):
            klines = fetch_binance_klines(
                ticker.symbol,
                longshort_strategy.candle_interval,
                last_kline_open_time_ms,
            )
        else:
            klines = pd.DataFrame()

        if not klines.empty:
            last_kline_open_time_sec = get_last_kline_open_time(klines)
            df = read_dataset_to_mem(engine, ticker.dataset_name)
            df = pd.concat([df, klines], ignore_index=True)
            df = df.align(klines, axis=1)[0]
            df.sort_values(by="kline_open_time", ascending=True, inplace=True)
            df = df.tail(longshort_strategy.num_req_klines + NUM_REQ_KLINES_BUFFER)

            convert_orig_cols_to_numeric(df)

            decisions, transformed_data = long_short_transform_and_predict(
                longshort_strategy, df, transformations
            )

            transformed_data.to_sql(
                ticker.dataset_name, con=engine, if_exists="replace", index=False
            )

            LongShortTickerQuery.update(
                ticker.id,
                {"last_kline_open_time_sec": last_kline_open_time_sec, **decisions},
            )
            return decisions

    return {
        "is_valid_buy": ticker.is_valid_buy,
        "is_valid_sell": ticker.is_valid_sell,
        "is_on_Pred_serv_err": ticker.is_on_pred_serv_err,
    }


def find_ticker(tickers, ticker_id):
    for item in tickers:
        if item.id == ticker_id:
            return item
    return None


def long_short_create_pairs(
    current_pairs,
    buy_candidates,
    sell_candidates,
    long_short_group_id,
    tickers,
    current_state_dict,
):
    used_buy_ids = {pair.buy_ticker_id for pair in current_pairs}
    used_sell_ids = {pair.sell_ticker_id for pair in current_pairs}

    used_ids = used_buy_ids | used_sell_ids

    valid_buys = [buy for buy in buy_candidates if buy not in used_ids]
    valid_sells = [sell for sell in sell_candidates if sell not in used_ids]

    n_valid_pairs = min(len(valid_buys), len(valid_sells))
    current_state_dict["total_available_pairs"] = n_valid_pairs
    exhausted_ids = set()

    for i in range(n_valid_pairs):
        buy_ticker_id = valid_buys[i]
        sell_ticker_id = None

        for j in range(n_valid_pairs):
            sell_ticker_id = valid_sells[j]
            if buy_ticker_id == sell_ticker_id or sell_ticker_id in exhausted_ids:
                continue
            else:
                break

        if sell_ticker_id is None:
            continue

        if (
            sell_ticker_id == buy_ticker_id
            or sell_ticker_id in exhausted_ids
            or buy_ticker_id in exhausted_ids
        ):
            continue

        buy_ticker = find_ticker(tickers, buy_ticker_id)
        sell_ticker = find_ticker(tickers, sell_ticker_id)

        if buy_ticker is None or sell_ticker is None:
            continue

        LongShortPairQuery.create_entry(
            {
                "long_short_group_id": long_short_group_id,
                "buy_ticker_id": buy_ticker_id,
                "sell_ticker_id": sell_ticker_id,
                "buy_ticker_dataset_name": buy_ticker.dataset_name,
                "sell_ticker_dataset_name": sell_ticker.dataset_name,
                "buy_symbol": buy_ticker.symbol,
                "buy_base_asset": buy_ticker.base_asset,
                "buy_quote_asset": buy_ticker.quote_asset,
                "sell_symbol": sell_ticker.symbol,
                "sell_base_asset": sell_ticker.base_asset,
                "sell_quote_asset": sell_ticker.quote_asset,
                "buy_qty_precision": buy_ticker.trade_quantity_precision,
                "sell_qty_precision": sell_ticker.trade_quantity_precision,
            }
        )

        exhausted_ids.add(sell_ticker_id)
        exhausted_ids.add(buy_ticker_id)


def long_short_process_pair_exit(longshort_strategy, pair):
    buy_df = read_latest_row(engine, pair.buy_ticker_dataset_name).iloc[0]
    sell_df = read_latest_row(engine, pair.sell_ticker_dataset_name).iloc[0]

    results_dict = {
        "buy_df": buy_df,
        "sell_df": sell_df,
        "should_close_trade": False,
    }

    helper = {"{EXIT_PAIR_TRADE_FUNC}": longshort_strategy.exit_cond}

    exec(
        replace_placeholders_on_code_templ(CodeTemplates.LONG_SHORT_PAIR_EXIT, helper),
        globals(),
        results_dict,
    )

    should_close_trade = results_dict["should_close_trade"]
    LongShortPairQuery.update_entry(pair.id, {"should_close": should_close_trade})


def find_ls_ticker_based_on_is(id, tickers):
    for item in tickers:
        if id == item.id:
            return item
    return None


def remove_is_no_loan_available_err_pair(pair, tickers, current_state_dict):
    if pair.is_no_loan_available_err is True and pair.in_position is False:
        long_side_ticker = find_ls_ticker_based_on_is(pair.buy_ticker_id, tickers)
        short_side_ticker = find_ls_ticker_based_on_is(pair.sell_ticker_id, tickers)
        current_state_dict["no_loan_available_err"] += 1

        if long_side_ticker is None or short_side_ticker is None:
            return

        if (
            long_side_ticker.is_valid_buy is False
            or short_side_ticker.is_valid_sell is False
        ):
            LongShortPairQuery.delete_entry(pair.id)


def update_long_short_exits(longshort_strategy, current_state_dict):
    with LogExceptionContext(re_raise=False):
        current_pairs = LongShortPairQuery.get_pairs_by_group_id(longshort_strategy.id)
        current_tickers = LongShortTickerQuery.get_all_by_group_id(
            longshort_strategy.id
        )

        for item in current_pairs:
            long_short_process_pair_exit(longshort_strategy, item)
            remove_is_no_loan_available_err_pair(
                item, current_tickers, current_state_dict
            )


def update_long_short_enters(longshort_strategy, current_state_dict):
    with LogExceptionContext(re_raise=False):
        tickers = LongShortTickerQuery.get_all_by_group_id(longshort_strategy.id)
        current_pairs = LongShortPairQuery.get_pairs_by_group_id(longshort_strategy.id)
        transformations = (
            DataTransformationQuery.get_transformations_by_longshort_group(
                longshort_strategy.id
            )
        )

        buy_candidates = set()
        sell_candidates = set()
        current_state_dict["total_num_symbols"] += len(tickers)

        for ticker in tickers:
            results = long_short_process_ticker(
                longshort_strategy, transformations, ticker
            )

            is_valid_buy = results["is_valid_buy"]
            is_valid_sell = results["is_valid_sell"]

            if is_valid_buy is True:
                current_state_dict["buy_symbols"].append(ticker.symbol)
                buy_candidates.add(ticker.id)

            if is_valid_sell is True:
                current_state_dict["sell_symbols"].append(ticker.symbol)
                sell_candidates.add(ticker.id)

        long_short_create_pairs(
            current_pairs,
            buy_candidates,
            sell_candidates,
            longshort_strategy.id,
            tickers,
            current_state_dict,
        )


def is_stop_loss_hit(prices: List[int], stop_loss_threshold, is_short_selling_strategy):
    if len(prices) == 0:
        return False
    if is_short_selling_strategy:
        price_ath = prices[0]

        for item in prices:
            if ((price_ath / item) - 1) * 100 > stop_loss_threshold:
                return True
            if item > price_ath:
                price_ath = item
    else:
        price_atl = prices[0]

        for item in prices:
            if ((item / price_atl) - 1) * 100 > stop_loss_threshold:
                return True
            if item < price_atl:
                price_atl = item
    return False


def is_take_profit_hit(
    last_price, open_price, take_profit_threshold, is_short_selling_strategy
):
    if is_short_selling_strategy:
        profit_percentage = ((open_price - last_price) / open_price) * 100
    else:
        profit_percentage = ((last_price - open_price) / open_price) * 100

    return profit_percentage >= take_profit_threshold


def update_trading_decisions_based_on_stops_v2(results_dict, local_dataset, last_price):
    if (
        local_dataset.strategy.is_in_position
        and local_dataset.strategy.should_calc_stops_on_pred_serv is True
    ):
        if results_dict["should_enter_trade"] is True:
            return

        if (
            local_dataset.strategy.use_profit_based_close is False
            and local_dataset.strategy.use_stop_loss_based_close is False
        ):
            return

        prices = []
        prices.append(local_dataset.strategy.price_on_trade_open)

        if local_dataset.active_trade is None:
            create_log(
                msg=f"Conflict: Strategy: {local_dataset.strategy.name}, symbol: {local_dataset.strategy.symbol} did not have active trade despite being in a position.",
                level=LogLevel.EXCEPTION,
            )
            return

        info_ticks = local_dataset.active_trade.info_ticks
        for item in info_ticks:
            prices.append(item.price)
        prices.append(last_price)

        is_short_selling = (
            True if local_dataset.strategy.is_short_selling_strategy else False
        )

        if (
            local_dataset.strategy.use_stop_loss_based_close is True
            and is_stop_loss_hit(
                prices,
                local_dataset.strategy.stop_loss_threshold_perc,
                is_short_selling,
            )
            is True
        ):
            results_dict["should_close_trade"] = True

        if (
            local_dataset.strategy.use_profit_based_close is True
            and is_take_profit_hit(
                last_price,
                local_dataset.strategy.price_on_trade_open,
                local_dataset.strategy.take_profit_threshold_perc,
                is_short_selling,
            )
            is True
        ):
            results_dict["should_close_trade"] = True


def update_trading_decisions_based_on_stops(results_dict, df, strategy):
    with LogExceptionContext(re_raise=False):
        if (
            strategy.should_calc_stops_on_pred_serv is False
            or strategy.price_on_trade_open is None
        ):
            return

        last_price = df.iloc[-1]["close_price"]

        if strategy.use_stop_loss_based_close:
            if strategy.is_short_selling_strategy is True:
                threshold = 1 + (strategy.stop_loss_threshold_perc / 100)
                if last_price >= strategy.price_on_trade_open * threshold:
                    results_dict["should_close_trade"] = True
            else:
                threshold = 1 - (strategy.stop_loss_threshold_perc / 100)
                if last_price <= strategy.price_on_trade_open * threshold:
                    results_dict["should_close_trade"] = True

        if strategy.use_profit_based_close:
            if strategy.is_short_selling_strategy is True:
                threshold = 1 - (strategy.take_profit_threshold_perc / 100)
                if last_price <= strategy.price_on_trade_open * threshold:
                    results_dict["should_close_trade"] = True
            else:
                threshold = 1 + (strategy.take_profit_threshold_perc / 100)
                if last_price >= strategy.price_on_trade_open * threshold:
                    results_dict["should_close_trade"] = True
    return False


def gen_trading_decisions(strategy, state_manager: RuleBasedLoopManager):
    with LogExceptionContext(re_raise=False):
        local_dataset = state_manager.datasets[strategy.name]
        if local_dataset.last_kline_open_time_sec is None:
            klines = fetch_binance_klines(
                strategy.symbol,
                strategy.candle_interval,
                calculate_timestamp_for_kline_fetch(
                    strategy.num_req_klines, strategy.kline_size_ms
                ),
            )

            last_kline_open_time_sec = get_last_kline_open_time(klines)
            results = transform_and_predict(strategy, klines, local_dataset)

            price = klines.iloc[-1]["close_price"]
            trading_state_dict = {
                **results,
                "last_kline_open_time_sec": last_kline_open_time_sec,
            }
            if strategy.is_in_position is True and strategy.active_trade_id is not None:
                last_kline_open_time_ms = last_kline_open_time_sec * 1000
                state_manager.add_trade_info_tick(
                    price, last_kline_open_time_ms, strategy.active_trade_id
                )

            state_manager.update_local_dataset(
                strategy=strategy,
                klines=klines,
                last_kline_open_time_sec=last_kline_open_time_sec,
                trading_state_dict=trading_state_dict,
            )
            return results

        last_kline_open_time_ms = (local_dataset.last_kline_open_time_sec * 1000) + 1
        current_time_ms = get_current_timestamp_ms()

        if current_time_ms >= last_kline_open_time_ms + strategy.kline_size_ms * 2:
            klines = fetch_binance_klines(
                strategy.symbol, strategy.candle_interval, int(last_kline_open_time_ms)
            )
        else:
            klines = pd.DataFrame()

        if not klines.empty:
            last_kline_open_time_sec = get_last_kline_open_time(klines)
            df = local_dataset.dataset
            df = pd.concat([df, klines], ignore_index=True)
            results = transform_and_predict(strategy, df, local_dataset)

            price = klines.iloc[-1]["close_price"]
            update_trading_decisions_based_on_stops_v2(results, local_dataset, price)

            df = df.tail(strategy.num_req_klines + NUM_REQ_KLINES_BUFFER)

            trading_state_dict = {
                **results,
                "last_kline_open_time_sec": last_kline_open_time_sec,
            }

            if strategy.is_in_position is True and strategy.active_trade_id is not None:
                price = df.iloc[-1]["close_price"]
                last_kline_open_time_ms = last_kline_open_time_sec * 1000
                state_manager.add_trade_info_tick(
                    price, last_kline_open_time_ms, strategy.active_trade_id
                )
            state_manager.update_local_dataset(
                strategy=strategy,
                klines=df,
                last_kline_open_time_sec=last_kline_open_time_sec,
                trading_state_dict=trading_state_dict,
            )
            return results

        return {
            "should_enter_trade": local_dataset.should_enter_trade,
            "should_close_trade": local_dataset.should_close_trade,
            "is_on_pred_serv_err": local_dataset.is_on_pred_serv_err,
        }


def infer_assets(symbol: str) -> dict:
    if symbol.endswith("USDT"):
        quote_asset = "USDT"
        base_asset = symbol[:-4]
    elif symbol.endswith("BTC"):
        quote_asset = "BTC"
        base_asset = symbol[:-3]
    elif symbol.endswith("ETH"):
        quote_asset = "ETH"
        base_asset = symbol[:-3]
    elif symbol.endswith("BNB"):
        quote_asset = "BNB"
        base_asset = symbol[:-3]
    else:
        raise ValueError("Unsupported quote asset.")

    return {"baseAsset": base_asset, "quoteAsset": quote_asset}
