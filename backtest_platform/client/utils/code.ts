import { CodeHelper } from "./constants";

export const ENTER_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def open_long_trade(tick):");
  code.addIndent();
  code.appendLine("return True");
  return code.get();
};

export const EXIT_LONG_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def close_long_trade(tick):");
  code.addIndent();
  code.appendLine("return True");
  return code.get();
};

export const EXIT_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def open_short_trade(tick):");
  code.addIndent();
  code.appendLine("return True");
  return code.get();
};

export const EXIT_SHORT_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def close_short_trade(tick):");
  code.addIndent();
  code.appendLine("return True");
  return code.get();
};

export const CREATE_COLUMNS_DEFAULT = () => {
  const code = new CodeHelper();
  code.appendLine("dataset = get_dataset()");

  return code.get();
};

export const ML_ENTER_TRADE_COND = () => {
  const code = new CodeHelper();

  code.appendLine("def get_enter_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction > 1.01");
  return code.get();
};

export const ML_EXIT_TRADE_COND = () => {
  const code = new CodeHelper();

  code.appendLine("def get_exit_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction < 0.99");
  return code.get();
};

export const DEPLOY_STRAT_ENTER_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def get_enter_trade_decision(transformed_df):");
  code.addIndent();
  code.appendLine("return False");
  return code.get();
};

export const DEPLOY_STRAT_EXIT_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def get_exit_trade_decision(transformed_df):");
  code.addIndent();
  code.appendLine("return False");
  return code.get();
};

export const FETCH_DATASOURCES_DEFAULT: string = `def fetch_datasources():
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
    return df`;

export const DATA_TRANSFORMATIONS_EXAMPLE: string = `def make_data_transformations(fetched_data):
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

    return fetched_data`;
