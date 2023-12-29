from enum import Enum

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

LOG_FILE = "logs"
DB_DATASETS = "datasets.db"
DB_DATASETS_UTIL = "datasets_util.db"
DB_LOGS = "logs.db"
DATASET_UTIL_TABLE_NAME = "dataset_util"


class DomEventChannels(Enum):
    REFETCH_ALL_DATASETS = "refetch_all_datasets"


class DatasetUtilsColumns(Enum):
    TABLE_NAME = "table_name"
    TIMESERIES_COLUMN = "timeseries_column"
