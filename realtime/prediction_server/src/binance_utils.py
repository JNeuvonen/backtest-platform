import sqlite3
import os
import pandas as pd
from binance import Client
from log import LogExceptionContext
from dataset_utils import read_dataset_to_mem
from code_gen_templates import CodeTemplates
from schema.data_transformation import (
    DataTransformationQuery,
)

from sqlite_schema.data_fetcher import DataFetcherQuery
from utils import (
    NUM_REQ_KLINES_BUFFER,
    calculate_timestamp_for_kline_fetch,
    gen_data_transformations_code,
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


def strategy_local_db_path(strategy_name):
    if not os.path.exists(DATABASES_DIR):
        os.makedirs(DATABASES_DIR)
    return os.path.join(DATABASES_DIR, strategy_name)


def cleanup_unused_disk_space(strategies):
    files = []

    for filename in os.listdir(DATABASES_DIR):
        file_path = os.path.join(DATABASES_DIR, filename)
        files.append(filename)

        if not any(strategy.name == filename for strategy in strategies):
            os.remove(file_path)


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
    return df


def update_last_kline_open_time(strategy_id, klines_df):
    if not klines_df.empty:
        last_kline_open_time_ms = klines_df.iloc[-1]["kline_open_time"]
        DataFetcherQuery.update_last_kline_open_time(
            strategy_id, last_kline_open_time_ms / 1000
        )


def transform_and_predict(strategy, df):
    data_transformations = DataTransformationQuery.get_transformations_by_strategy(
        strategy.id
    )

    data_transformations_helper = {
        "{DATA_TRANSFORMATIONS_FUNC}": gen_data_transformations_code(
            data_transformations
        )
    }
    results_dict = {
        "transformed_data": None,
        "fetched_data": df,
        "should_enter_trade": None,
        "should_exit_trade": None,
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


def gen_trading_decisions(strategy):
    with LogExceptionContext(re_raise=False):
        data_fetcher = DataFetcherQuery.get_by_strat_id(strategy.id)

        if data_fetcher is None:
            raise Exception("data_fetcher was unexpectedly None")

        if data_fetcher.last_kline_open_time_sec is None:
            klines = fetch_binance_klines(
                strategy.symbol,
                strategy.candle_interval,
                calculate_timestamp_for_kline_fetch(
                    strategy.num_req_klines, strategy.kline_size_ms
                ),
            )
            with sqlite3.connect(strategy_local_db_path(strategy.name)) as conn:
                klines.to_sql(strategy.name, conn, if_exists="replace", index=False)

            update_last_kline_open_time(strategy.id, klines)
            results = transform_and_predict(strategy, klines)
            DataFetcherQuery.update_trade_flags(
                strategy.id,
                results["should_enter_trade"],
                results["should_close_trade"],
                results["is_on_pred_serv_err"],
            )
            return results

        last_kline_open_time_ms = (data_fetcher.last_kline_open_time_sec * 1000) + 1
        klines = fetch_binance_klines(
            strategy.symbol, strategy.candle_interval, last_kline_open_time_ms
        )

        if not klines.empty:
            update_last_kline_open_time(strategy.id, klines)
            df = read_dataset_to_mem(
                strategy_local_db_path(strategy.name), strategy.name
            )
            df = df.append(klines, ignore_index=True)
            df = df.align(klines, axis=1)[0]
            df.sort_values(by="kline_open_time", ascending=True, inplace=True)

            results = transform_and_predict(strategy, df)

            df = df.tail(strategy.num_req_klines + NUM_REQ_KLINES_BUFFER)

            with sqlite3.connect(strategy_local_db_path(strategy.name)) as conn:
                df.to_sql(strategy.name, conn, if_exists="replace", index=False)

            DataFetcherQuery.update_trade_flags(
                strategy.id,
                results["should_enter_trade"],
                results["should_close_trade"],
                results["is_on_pred_serv_err"],
            )
            return results

        return {
            "should_enter_trade": data_fetcher.prev_should_open_trade,
            "should_close_trade": data_fetcher.prev_should_close_trade,
            "is_on_pred_serv_err": data_fetcher.is_on_pred_serv_err,
        }
