import pandas as pd
from binance import Client


client = Client()

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


def fetch_binance_klines(symbol, interval, start_str):
    klines = client.get_historical_klines(
        symbol=symbol, interval=interval, start_str=start_str
    )

    klines = klines[:-1] if klines else klines
    df = pd.DataFrame(klines, columns=BINANCE_DATA_COLS)
    return df
