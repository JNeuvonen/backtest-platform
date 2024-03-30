from t_constants import ONE_DAY_IN_MS


class TradingRules:
    class RSI_30_MA_50_CLOSE_PRICE:
        OPEN = """
def get_enter_trade_decision(transformed_df):
    if transformed_df is None or transformed_df.empty: 
        return False
    tick = transformed_df.iloc[len(df) - 1]
    return tick["RSI_30_MA_50_close_price"] > 95
"""

        CLOSE = """
def get_exit_trade_decision(transformed_df):
    if transformed_df is None or transformed_df.empty: 
        return False
    tick = transformed_df.iloc[len(df) - 1]
    return tick["RSI_30_MA_50_close_price"] < 95
"""

        FETCH = """
def fetch_datasources():
    import pandas as pd
    from binance import Client
    def get_historical_klines(symbol, interval):
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
        client = Client()
        start_time = "1 Jan, 2017"
        klines = []

        while True:
            new_klines = client.get_historical_klines(
                symbol, interval, start_time, limit=1000
            )
            if not new_klines:
                break

            klines.extend(new_klines)
            start_time = int(new_klines[-1][0]) + 1

        df = pd.DataFrame(klines, columns=BINANCE_DATA_COLS)
        df.drop(["ignore", "kline_close_time"], axis=1, inplace=True)
        df["kline_open_time"] = pd.to_numeric(df["kline_open_time"])

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.sort_values("kline_open_time", inplace=True)
        return df

    df = get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY)
    return df
"""

        TRANSFORMATIONS = """
def make_data_transformations(fetched_data):
    def calculate_ma(df, column="close_price", periods=[50]):
        for period in periods:
            ma_label = f"MA_{period}_{column}"
            df[ma_label] = df[column].rolling(window=period).mean()

    periods = [50]
    column = "close_price"
    calculate_ma(fetched_data, column=column, periods=periods)

    def calculate_rsi(df, column="open_price", periods=[14]):
        for period in periods:
            delta = df[column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            df_label = f"RSI_{period}_{column}"
            df[df_label] = 100 - (100 / (1 + rs))

    periods = [30]
    column = "MA_50_close_price"
    calculate_rsi(fetched_data, column=column, periods=periods)
    return fetched_data
"""


def create_strategy_body(
    symbol: str,
    enter_trade_code: str,
    exit_trade_code: str,
    fetch_datasources_code: str,
    data_transformations_code: str,
    priority: int,
    kline_size_ms: int,
    klines_left_till_autoclose: int,
    minimum_time_between_trades_ms: int,
    allocated_size_perc: float,
    take_profit_threshold_perc: float,
    stop_loss_threshold_perc: float,
    use_testnet: bool,
    use_time_based_close: bool,
    use_profit_based_close: bool,
    use_stop_loss_based_close: bool,
    use_taker_order: bool,
    is_leverage_allowed: bool,
    is_short_selling_strategy: bool,
):
    return {
        "symbol": symbol,
        "enter_trade_code": enter_trade_code,
        "exit_trade_code": exit_trade_code,
        "fetch_datasources_code": fetch_datasources_code,
        "data_transformations_code": data_transformations_code,
        "priority": priority,
        "kline_size_ms": kline_size_ms,
        "klines_left_till_autoclose": klines_left_till_autoclose,
        "minimum_time_between_trades_ms": minimum_time_between_trades_ms,
        "allocated_size_perc": allocated_size_perc,
        "take_profit_threshold_perc": take_profit_threshold_perc,
        "stop_loss_threshold_perc": stop_loss_threshold_perc,
        "use_testnet": use_testnet,
        "use_time_based_close": use_time_based_close,
        "use_profit_based_close": use_profit_based_close,
        "use_stop_loss_based_close": use_stop_loss_based_close,
        "use_taker_order": use_taker_order,
        "is_leverage_allowed": is_leverage_allowed,
        "is_short_selling_strategy": is_short_selling_strategy,
    }


def strategy_simple_1():
    return create_strategy_body(
        symbol="BTCUSDT",
        enter_trade_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.OPEN,
        exit_trade_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.CLOSE,
        fetch_datasources_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.FETCH,
        data_transformations_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.TRANSFORMATIONS,
        priority=1,
        kline_size_ms=ONE_DAY_IN_MS,
        klines_left_till_autoclose=0,
        allocated_size_perc=100,
        take_profit_threshold_perc=0,
        stop_loss_threshold_perc=0,
        minimum_time_between_trades_ms=1000,
        use_testnet=True,
        use_time_based_close=False,
        use_profit_based_close=False,
        use_stop_loss_based_close=False,
        use_taker_order=False,
        is_leverage_allowed=False,
        is_short_selling_strategy=False,
    )
