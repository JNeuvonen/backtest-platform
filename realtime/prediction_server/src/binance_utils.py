import pandas as pd
from binance import Client
from log import LogExceptionContext
from orm import engine
from dataset_utils import read_dataset_to_mem
from code_gen_templates import CodeTemplates
from schema.data_transformation import (
    DataTransformationQuery,
)
from schema.strategy import StrategyQuery

from utils import (
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


def transform_and_predict(strategy, df):
    data_transformations = DataTransformationQuery.get_transformations_by_strategy(
        strategy.id
    )

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


def gen_trading_decisions(strategy):
    with LogExceptionContext(re_raise=False):
        if strategy.last_kline_open_time_sec is None:
            klines = fetch_binance_klines(
                strategy.symbol,
                strategy.candle_interval,
                calculate_timestamp_for_kline_fetch(
                    strategy.num_req_klines, strategy.kline_size_ms
                ),
            )
            klines.to_sql(strategy.name, con=engine, if_exists="replace", index=False)

            last_kline_open_time_sec = get_last_kline_open_time(klines)
            results = transform_and_predict(strategy, klines)
            StrategyQuery.update_trade_flags(
                strategy.id,
                results["should_enter_trade"],
                results["should_close_trade"],
                results["is_on_pred_serv_err"],
                last_kline_open_time_sec,
            )
            return results

        last_kline_open_time_ms = (strategy.last_kline_open_time_sec * 1000) + 1
        current_time_ms = get_current_timestamp_ms()

        if current_time_ms >= last_kline_open_time_ms + strategy.kline_size_ms * 2:
            klines = fetch_binance_klines(
                strategy.symbol, strategy.candle_interval, last_kline_open_time_ms
            )
        else:
            klines = pd.DataFrame()

        if not klines.empty:
            last_kline_open_time_sec = get_last_kline_open_time(klines)
            df = read_dataset_to_mem(engine, strategy.name)
            df = pd.concat([df, klines], ignore_index=True)
            df = df.align(klines, axis=1)[0]
            df.sort_values(by="kline_open_time", ascending=True, inplace=True)

            convert_orig_cols_to_numeric(df)

            results = transform_and_predict(strategy, df)

            df = df.tail(strategy.num_req_klines + NUM_REQ_KLINES_BUFFER)

            df.to_sql(strategy.name, con=engine, if_exists="replace", index=False)

            StrategyQuery.update_trade_flags(
                strategy.id,
                results["should_enter_trade"],
                results["should_close_trade"],
                results["is_on_pred_serv_err"],
                last_kline_open_time_sec,
            )
            return results

        return {
            "should_enter_trade": strategy.should_enter_trade,
            "should_close_trade": strategy.should_close_trade,
            "is_on_pred_serv_err": strategy.is_on_pred_serv_err,
        }
