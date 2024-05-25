import datetime
from common_python.constants import ONE_DAY_IN_MS


transformation_id = 0

TRANSFORMATION_1 = """
def calculate_ma(df, column='close_price', periods=[50]):
    for period in periods:
        ma_label = f"MA_{period}_{column}"
        df[ma_label] = df[column].rolling(window=period).mean()

periods = [50]
column = "close_price"
calculate_ma(dataset, column=column, periods=periods)
"""


TRANSFORMATION_2 = """
def calculate_rsi(df, column='open_price', periods=[14]):
    for period in periods:
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df_label = f"RSI_{period}_{column}"
        df[df_label] = 100 - (100 / (1 + rs))

periods = [30]
column = "MA_50_close_price"
calculate_rsi(dataset, column=column, periods=periods)
"""


def gen_data_transformation_dict(transformation_str):
    global transformation_id
    transformation_id += 1
    current_time = datetime.datetime.now()
    return {
        "id": transformation_id,
        "transformation_code": transformation_str,
        "created_at": current_time.isoformat(),
        "updated_at": current_time.isoformat(),
    }


class TradingRules:
    class RSI_30_MA_50_CLOSE_PRICE:
        OPEN = """
def get_enter_trade_decision(transformed_df):
    if transformed_df is None or transformed_df.empty: 
        return False
    tick = transformed_df.iloc[len(transformed_df) - 1]
    return tick["RSI_30_MA_50_close_price"] > 95
"""

        CLOSE = """
def get_exit_trade_decision(transformed_df):
    if transformed_df is None or transformed_df.empty: 
        return False
    tick = transformed_df.iloc[len(transformed_df) - 1]
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

    class ShortSimple:
        OPEN = """
def get_enter_trade_decision(transformed_df):
    if transformed_df is None or transformed_df.empty: 
        return False
    tick = transformed_df.iloc[len(transformed_df) - 1]
    return tick["RSI_100_MA_720_OBV"] < 10 and tick["RSI_160_MA_200_close_price"] < 10 and tick["close_price"] > tick["MA_200_close_price"]
"""

        CLOSE = """
def get_exit_trade_decision(transformed_df):
    if transformed_df is None or transformed_df.empty: 
        return False
    tick = transformed_df.iloc[len(transformed_df) - 1]
    return tick["RSI_100_MA_720_OBV"] > 30
"""

        TRANSFORMATIONS = """
def make_data_transformations(fetched_data):
    def calculate_obv(df):
        df['OBV'] = 0
        for i in range(1, len(df)):
            if df['open_price'][i] > df['open_price'][i-1]:
                df['OBV'][i] = df['OBV'][i-1] + df['volume'][i]
            elif df['open_price'][i] < df['open_price'][i-1]:
                df['OBV'][i] = df['OBV'][i-1] - df['volume'][i]
            else:
                df['OBV'][i] = df['OBV'][i-1]

    def calculate_ma(df, column="close_price", periods=[50]):
        for period in periods:
            ma_label = f"MA_{period}_{column}"
            df[ma_label] = df[column].rolling(window=period).mean()


    def calculate_rsi(df, column="open_price", periods=[14]):
        for period in periods:
            delta = df[column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            df_label = f"RSI_{period}_{column}"
            df[df_label] = 100 - (100 / (1 + rs))

    
    calculate_obv(fetched_data)

    periods = [720]
    column = "OBV"
    calculate_ma(fetched_data, column=column, periods=periods)

    periods = [100]
    column = "MA_720_OBV"
    calculate_rsi(fetched_data, column=column, periods=periods)


    periods = [200]
    column = "close_price"
    calculate_ma(fetched_data, column=column, periods=periods)

    periods = [160]
    column = "MA_200_close_price"
    calculate_rsi(fetched_data, column=column, periods=periods)

    return fetched_data
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

    df = get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR)
    return df
"""

    class ShortDevPurposes:
        OPEN = """
def get_enter_trade_decision(transformed_df):
    return True 
"""

        CLOSE = """
def get_exit_trade_decision(transformed_df):
    return False 
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

    df = get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR)
    return df
"""

        TRANSFORMATIONS = """
def make_data_transformations(fetched_data):
    #debugging strategy - no need to actually transform the data
    return fetched_data
"""

    class StratWithSyntaxErr:
        OPEN = """
def get_enter_trade_decision(transformed_df):
    return False 
"""

        CLOSE = """
def get_exit_trade_decision(transformed_df):
    return False 
"""

        FETCH = """
def fetch_datasources():
    import pandas as pd
    from binance import Client
    #syntax error
    prin(hello world)
"""

        TRANSFORMATIONS = """
def make_data_transformations(fetched_data):
    #debugging strategy - no need to actually transform the data
    return fetched_data
"""


def create_strategy_body(
    name: str,
    symbol: str,
    base_asset: str,
    quote_asset: str,
    enter_trade_code: str,
    exit_trade_code: str,
    fetch_datasources_code: str,
    data_transformations_code: str,
    trade_quantity_precision: int,
    priority: int,
    kline_size_ms: int,
    minimum_time_between_trades_ms: int,
    maximum_klines_hold_time: int,
    allocated_size_perc: float,
    take_profit_threshold_perc: float,
    stop_loss_threshold_perc: float,
    use_time_based_close: bool,
    num_req_klines: int,
    use_profit_based_close: bool,
    use_stop_loss_based_close: bool,
    use_taker_order: bool,
    is_leverage_allowed: bool,
    is_short_selling_strategy: bool,
    is_paper_trade_mode: bool,
    data_transformations=[],
):
    return {
        "name": name,
        "symbol": symbol,
        "base_asset": base_asset,
        "quote_asset": quote_asset,
        "enter_trade_code": enter_trade_code,
        "exit_trade_code": exit_trade_code,
        "fetch_datasources_code": fetch_datasources_code,
        "data_transformations_code": data_transformations_code,
        "trade_quantity_precision": trade_quantity_precision,
        "priority": priority,
        "kline_size_ms": kline_size_ms,
        "minimum_time_between_trades_ms": minimum_time_between_trades_ms,
        "maximum_klines_hold_time": maximum_klines_hold_time,
        "allocated_size_perc": allocated_size_perc,
        "take_profit_threshold_perc": take_profit_threshold_perc,
        "stop_loss_threshold_perc": stop_loss_threshold_perc,
        "use_time_based_close": use_time_based_close,
        "use_profit_based_close": use_profit_based_close,
        "use_stop_loss_based_close": use_stop_loss_based_close,
        "num_req_klines": num_req_klines,
        "use_taker_order": use_taker_order,
        "is_leverage_allowed": is_leverage_allowed,
        "is_short_selling_strategy": is_short_selling_strategy,
        "is_paper_trade_mode": is_paper_trade_mode,
        "data_transformations": data_transformations,
    }


def strategy_simple_1():
    return create_strategy_body(
        name="SimpleLongStrat",
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        enter_trade_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.OPEN,
        exit_trade_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.CLOSE,
        fetch_datasources_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.FETCH,
        data_transformations_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.TRANSFORMATIONS,
        trade_quantity_precision=5,
        priority=1,
        kline_size_ms=ONE_DAY_IN_MS,
        maximum_klines_hold_time=0,
        allocated_size_perc=100,
        take_profit_threshold_perc=0,
        stop_loss_threshold_perc=0,
        minimum_time_between_trades_ms=1000,
        num_req_klines=100,
        use_time_based_close=False,
        use_profit_based_close=False,
        use_stop_loss_based_close=False,
        use_taker_order=False,
        is_leverage_allowed=False,
        is_short_selling_strategy=False,
        is_paper_trade_mode=False,
        data_transformations=[
            gen_data_transformation_dict(TRANSFORMATION_1),
            gen_data_transformation_dict(TRANSFORMATION_2),
        ],
    )


def create_short_strategy_simple_1():
    return create_strategy_body(
        name="SimpleShortStrat",
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        enter_trade_code=TradingRules.ShortSimple.OPEN,
        exit_trade_code=TradingRules.ShortSimple.CLOSE,
        fetch_datasources_code=TradingRules.ShortSimple.FETCH,
        data_transformations_code=TradingRules.ShortSimple.TRANSFORMATIONS,
        trade_quantity_precision=5,
        priority=1,
        kline_size_ms=ONE_DAY_IN_MS,
        maximum_klines_hold_time=0,
        allocated_size_perc=50,
        take_profit_threshold_perc=0,
        stop_loss_threshold_perc=2,
        minimum_time_between_trades_ms=1000,
        use_time_based_close=False,
        use_profit_based_close=False,
        use_stop_loss_based_close=True,
        use_taker_order=False,
        is_leverage_allowed=False,
        is_short_selling_strategy=True,
        is_paper_trade_mode=False,
        data_transformations=[
            gen_data_transformation_dict(TRANSFORMATION_1),
            gen_data_transformation_dict(TRANSFORMATION_2),
        ],
    )


def create_short_strategy_simple_2():
    return create_strategy_body(
        name="ShortDevPurposes",
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        enter_trade_code=TradingRules.ShortDevPurposes.OPEN,
        exit_trade_code=TradingRules.ShortDevPurposes.CLOSE,
        fetch_datasources_code=TradingRules.ShortDevPurposes.FETCH,
        data_transformations_code=TradingRules.ShortDevPurposes.TRANSFORMATIONS,
        trade_quantity_precision=5,
        priority=1,
        kline_size_ms=ONE_DAY_IN_MS,
        allocated_size_perc=50,
        maximum_klines_hold_time=24,
        take_profit_threshold_perc=0,
        stop_loss_threshold_perc=2,
        minimum_time_between_trades_ms=1000,
        use_time_based_close=False,
        use_profit_based_close=False,
        use_stop_loss_based_close=True,
        use_taker_order=False,
        is_leverage_allowed=False,
        is_short_selling_strategy=True,
        is_paper_trade_mode=False,
        data_transformations=[
            gen_data_transformation_dict(TRANSFORMATION_1),
            gen_data_transformation_dict(TRANSFORMATION_2),
        ],
    )


def create_strategy_with_syntax_err():
    return create_strategy_body(
        name="SyntaxError",
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        enter_trade_code=TradingRules.StratWithSyntaxErr.OPEN,
        exit_trade_code=TradingRules.StratWithSyntaxErr.CLOSE,
        fetch_datasources_code=TradingRules.StratWithSyntaxErr.FETCH,
        data_transformations_code=TradingRules.StratWithSyntaxErr.TRANSFORMATIONS,
        trade_quantity_precision=5,
        priority=1,
        kline_size_ms=ONE_DAY_IN_MS,
        allocated_size_perc=50,
        maximum_klines_hold_time=24,
        take_profit_threshold_perc=2,
        stop_loss_threshold_perc=2,
        minimum_time_between_trades_ms=1000,
        use_time_based_close=False,
        use_profit_based_close=False,
        use_stop_loss_based_close=True,
        use_taker_order=False,
        is_leverage_allowed=False,
        is_short_selling_strategy=True,
        is_paper_trade_mode=False,
        data_transformations=[
            gen_data_transformation_dict(TRANSFORMATION_1),
            gen_data_transformation_dict(TRANSFORMATION_2),
        ],
    )


def update_strat_on_trade_open_test_case_1(trade_id: int, strat_id: int):
    return {
        "active_trade_id": trade_id,
        "id": strat_id,
        "price_on_trade_open": 65000,
        "time_on_trade_open_ms": 1712340861903,
    }


test_case_dump = {
    "name": "testtt",
    "symbol": "BTCUSDT",
    "base_asset": "BTC",
    "quote_asset": "USDT",
    "enter_trade_code": 'def get_enter_trade_decision(transformed_df):\n    return transformed_df["RSI_160_MA_200_close_price"] > 95',
    "exit_trade_code": 'def get_exit_trade_decision(transformed_df):\n    return transformed_df["RSI_160_MA_200_close_price"] < 95',
    "fetch_datasources_code": 'def fetch_datasources():\n    import pandas as pd\n    from binance import Client\n    def get_historical_klines(symbol, interval):\n        BINANCE_DATA_COLS = [\n            "kline_open_time",\n            "open_price",\n            "high_price",\n            "low_price",\n            "close_price",\n            "volume",\n            "kline_close_time",\n            "quote_asset_volume",\n            "number_of_trades",\n            "taker_buy_base_asset_volume",\n            "taker_buy_quote_asset_volume",\n            "ignore",\n        ]\n        client = Client()\n        start_time = "1 Jan, 2017"\n        klines = []\n\n        while True:\n            new_klines = client.get_historical_klines(\n                symbol, interval, start_time, limit=1000\n            )\n            if not new_klines:\n                break\n\n            klines.extend(new_klines)\n            start_time = int(new_klines[-1][0]) + 1\n\n        df = pd.DataFrame(klines, columns=BINANCE_DATA_COLS)\n        df.drop(["ignore", "kline_close_time"], axis=1, inplace=True)\n        df["kline_open_time"] = pd.to_numeric(df["kline_open_time"])\n\n        for col in df.columns:\n            df[col] = pd.to_numeric(df[col], errors="coerce")\n\n        df.sort_values("kline_open_time", inplace=True)\n        return df\n\n    df = get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR)\n    return df',
    "data_transformations_code": "def make_data_transformations(fetched_data):\n    def calculate_obv(df):\n        df['OBV'] = 0\n        for i in range(1, len(df)):\n            if df['open_price'][i] > df['open_price'][i-1]:\n                df['OBV'][i] = df['OBV'][i-1] + df['volume'][i]\n            elif df['open_price'][i] < df['open_price'][i-1]:\n                df['OBV'][i] = df['OBV'][i-1] - df['volume'][i]\n            else:\n                df['OBV'][i] = df['OBV'][i-1]\n\n    def calculate_ma(df, column=\"close_price\", periods=[50]):\n        for period in periods:\n            ma_label = f\"MA_{period}_{column}\"\n            df[ma_label] = df[column].rolling(window=period).mean()\n\n    def calculate_rsi(df, column=\"open_price\", periods=[14]):\n        for period in periods:\n            delta = df[column].diff()\n            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()\n            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()\n\n            rs = gain / loss\n            df_label = f\"RSI_{period}_{column}\"\n            df[df_label] = 100 - (100 / (1 + rs))\n\n    calculate_obv(fetched_data)\n\n    periods = [720]\n    column = \"OBV\"\n    calculate_ma(fetched_data, column=column, periods=periods)\n\n    periods = [100]\n    column = \"MA_720_OBV\"\n    calculate_rsi(fetched_data, column=column, periods=periods)\n\n    periods = [200]\n    column = \"close_price\"\n    calculate_ma(fetched_data, column=column, periods=periods)\n\n    periods = [160]\n    column = \"MA_200_close_price\"\n    calculate_rsi(fetched_data, column=column, periods=periods)\n\n    return fetched_data",
    "trade_quantity_precision": 5,
    "priority": 1,
    "kline_size_ms": 86400000,
    "maximum_klines_hold_time": 30,
    "allocated_size_perc": 100,
    "take_profit_threshold_perc": 20,
    "stop_loss_threshold_perc": 2,
    "minimum_time_between_trades_ms": 0,
    "use_time_based_close": False,
    "use_profit_based_close": False,
    "use_stop_loss_based_close": False,
    "use_taker_order": True,
    "is_leverage_allowed": False,
    "is_short_selling_strategy": False,
    "is_paper_trade_mode": False,
    "candle_interval": "1d",
    "num_req_klines": 700,
    "data_transformations": [
        {
            "id": 1,
            "dataset_id": 5,
            "created_at": "2024-04-15T08:12:39",
            "updated_at": "2024-04-15T08:12:39",
            "transformation_code": 'def calculate_ma(df, column=\'close_price\', periods=[50]):\n    for period in periods:\n        ma_label = f"MA_{period}_{column}"\n        df[ma_label] = df[column].rolling(window=period).mean()\n\nperiods = [260]\ncolumn = "close_price"\ncalculate_ma(dataset, column=column, periods=periods)\n',
        },
        {
            "id": 2,
            "dataset_id": 5,
            "created_at": "2024-04-15T08:12:51",
            "updated_at": "2024-04-15T08:12:51",
            "transformation_code": 'def calculate_ma(df, column=\'close_price\', periods=[50]):\n    for period in periods:\n        ma_label = f"MA_{period}_{column}"\n        df[ma_label] = df[column].rolling(window=period).mean()\n\nperiods = [200]\ncolumn = "close_price"\ncalculate_ma(dataset, column=column, periods=periods)\n',
        },
        {
            "id": 3,
            "dataset_id": 5,
            "created_at": "2024-04-15T08:13:15",
            "updated_at": "2024-04-15T08:13:15",
            "transformation_code": 'def calculate_rsi(df, column=\'open_price\', periods=[14]):\n    for period in periods:\n        delta = df[column].diff()\n        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()\n        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()\n\n        rs = gain / loss\n        df_label = f"RSI_{period}_{column}"\n        df[df_label] = 100 - (100 / (1 + rs))\n\nperiods = [160]\ncolumn = "MA_200_close_price"\ncalculate_rsi(dataset, column=column, periods=periods)\n',
        },
    ],
}


def gen_test_case_dump_body():
    return test_case_dump
