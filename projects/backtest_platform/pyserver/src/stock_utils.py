import pandas as pd
from db import create_connection
from log import LogExceptionContext
from constants import AppConstants, BinanceDataCols
from query_dataset import DatasetQuery
import yfinance as yf


class YFinanceColumns:
    DATE = "Date"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"
    DIVIDENDS = "Dividends"
    STOCK_SPLITS = "Stock Splits"


def get_yfinance_table_name(symbol: str):
    return f"{symbol}_1d_yfinance"


def get_nyse_symbols():
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    nasdaq_symbols = pd.read_csv(url, sep="|")

    nasdaq_symbols = nasdaq_symbols[nasdaq_symbols["Financial Status"] == "N"]
    return nasdaq_symbols["Symbol"].tolist()


def fetch_stock_symbol_yfinance_data(symbol: str) -> pd.DataFrame:
    symbol = yf.Ticker(symbol)
    historical_data = symbol.history(period="max", interval="1d")
    return historical_data


def save_yfinance_historical_klines(symbol):
    with LogExceptionContext(notification_duration=60000):
        symbol_df = fetch_stock_symbol_yfinance_data(symbol)
        symbol_df.reset_index(inplace=True)
        symbol_df[YFinanceColumns.DATE] = pd.to_datetime(
            symbol_df[YFinanceColumns.DATE]
        )
        symbol_df[YFinanceColumns.DATE] = symbol_df[YFinanceColumns.DATE].apply(
            lambda x: int(x.timestamp() * 1000)
        )

        symbol_df = symbol_df.rename(
            columns={
                YFinanceColumns.DATE: BinanceDataCols.KLINE_OPEN_TIME,
                YFinanceColumns.CLOSE: BinanceDataCols.CLOSE_PRICE,
                YFinanceColumns.OPEN: BinanceDataCols.OPEN_PRICE,
                YFinanceColumns.HIGH: BinanceDataCols.HIGH_PRICE,
                YFinanceColumns.LOW: BinanceDataCols.LOW_PRICE,
                YFinanceColumns.VOLUME: BinanceDataCols.VOLUME,
            }
        )

        table_name = get_yfinance_table_name(symbol)

        table_exists_query = (
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )

        datasets_conn = create_connection(AppConstants.DB_DATASETS)
        cursor = datasets_conn.cursor()
        cursor.execute(table_exists_query)
        if cursor.fetchone():
            DatasetQuery.delete_entry_by_dataset_name(table_name)

        symbol_df.to_sql(table_name, datasets_conn, if_exists="replace", index=False)

        DatasetQuery.create_entry_from_dict(
            {
                "dataset_name": table_name,
                "timeseries_column": BinanceDataCols.KLINE_OPEN_TIME,
                "price_column": BinanceDataCols.CLOSE_PRICE,
                "symbol": symbol,
                "interval": "1d",
                "is_stockmarket_dataset": True,
            }
        )
