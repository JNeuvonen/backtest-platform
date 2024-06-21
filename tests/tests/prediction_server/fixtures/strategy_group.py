test_case_dump = {
    "name": "testtt",
    "symbol": "BTCUSDT",
    "base_asset": "BTC",
    "should_calc_stops_on_pred_serv": False,
    "quote_asset": "USDT",
    "enter_trade_code": 'def get_enter_trade_decision(transformed_df):\n    return transformed_df["RSI_160_MA_200_close_price"] > 95',
    "exit_trade_code": 'def get_exit_trade_decision(transformed_df):\n    return transformed_df["RSI_160_MA_200_close_price"] < 95',
    "fetch_datasources_code": 'def fetch_datasources():\n    import pandas as pd\n    from binance import Client\n    def get_historical_klines(symbol, interval):\n        BINANCE_DATA_COLS = [\n            "kline_open_time",\n            "open_price",\n            "high_price",\n            "low_price",\n            "close_price",\n            "volume",\n            "kline_close_time",\n            "quote_asset_volume",\n            "number_of_trades",\n            "taker_buy_base_asset_volume",\n            "taker_buy_quote_asset_volume",\n            "ignore",\n        ]\n        client = Client()\n        start_time = "1 Jan, 2017"\n        klines = []\n\n        while True:\n            new_klines = client.get_historical_klines(\n                symbol, interval, start_time, limit=1000\n            )\n            if not new_klines:\n                break\n\n            klines.extend(new_klines)\n            start_time = int(new_klines[-1][0]) + 1\n\n        df = pd.DataFrame(klines, columns=BINANCE_DATA_COLS)\n        df.drop(["ignore", "kline_close_time"], axis=1, inplace=True)\n        df["kline_open_time"] = pd.to_numeric(df["kline_open_time"])\n\n        for col in df.columns:\n            df[col] = pd.to_numeric(df[col], errors="coerce")\n\n        df.sort_values("kline_open_time", inplace=True)\n        return df\n\n    df = get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR)\n    return df',
    "data_transformations_code": "def make_data_transformations(fetched_data):\n    def calculate_obv(df):\n        df['OBV'] = 0\n        for i in range(1, len(df)):\n            if df['open_price'][i] > df['open_price'][i-1]:\n                df['OBV'][i] = df['OBV'][i-1] + df['volume'][i]\n            elif df['open_price'][i] < df['open_price'][i-1]:\n                df['OBV'][i] = df['OBV'][i-1] - df['volume'][i]\n            else:\n                df['OBV'][i] = df['OBV'][i-1]\n\n    def calculate_ma(df, column=\"close_price\", periods=[50]):\n        for period in periods:\n            ma_label = f\"MA_{period}_{column}\"\n            df[ma_label] = df[column].rolling(window=period).mean()\n\n    def calculate_rsi(df, column=\"open_price\", periods=[14]):\n        for period in periods:\n            delta = df[column].diff()\n            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()\n            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()\n\n            rs = gain / loss\n            df_label = f\"RSI_{period}_{column}\"\n            df[df_label] = 100 - (100 / (1 + rs))\n\n    calculate_obv(fetched_data)\n\n    periods = [720]\n    column = \"OBV\"\n    calculate_ma(fetched_data, column=column, periods=periods)\n\n    periods = [100]\n    column = \"MA_720_OBV\"\n    calculate_rsi(fetched_data, column=column, periods=periods)\n\n    periods = [200]\n    column = \"close_price\"\n    calculate_ma(fetched_data, column=column, periods=periods)\n\n    periods = [160]\n    column = \"MA_200_close_price\"\n    calculate_rsi(fetched_data, column=column, periods=periods)\n\n    return fetched_data",
    "trade_quantity_precision": 5,
    "priority": 1,
    "kline_size_ms": 60000,
    "maximum_klines_hold_time": 30,
    "allocated_size_perc": 100,
    "take_profit_threshold_perc": 20,
    "num_symbols_for_auto_adaptive": 25,
    "stop_loss_threshold_perc": 2,
    "minimum_time_between_trades_ms": 0,
    "use_time_based_close": False,
    "use_profit_based_close": False,
    "use_stop_loss_based_close": False,
    "use_taker_order": True,
    "is_auto_adaptive_group": True,
    "is_leverage_allowed": False,
    "is_short_selling_strategy": False,
    "is_paper_trade_mode": False,
    "candle_interval": "1m",
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


test_case_dump["strategy_group"] = "test_strategy_group"


symbols = [
    {
        "symbol": "BTCUSDT",
        "base_asset": "BTC",
        "quote_asset": "USDT",
        "trade_quantity_precision": 3,
    },
    {
        "symbol": "ETHUSDT",
        "base_asset": "ETH",
        "quote_asset": "USDT",
        "trade_quantity_precision": 3,
    },
    {
        "symbol": "IOTAUSDT",
        "base_asset": "IOTA",
        "quote_asset": "USDT",
        "trade_quantity_precision": 3,
    },
    {
        "symbol": "PLTRUSDT",
        "base_asset": "PLTR",
        "quote_asset": "USDT",
        "trade_quantity_precision": 3,
    },
]

test_case_dump["symbols"] = symbols
