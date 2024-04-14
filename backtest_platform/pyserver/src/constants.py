from enum import Enum
from config import append_app_data_path

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

STREAMING_DEFAULT_CHUNK_SIZE = 1024

DATASET_UTILS_DB_PATH = "datasets_util.db"

ALEMBIC_VERSIONS_TAB = "alembic_version"


class DomEventChannels(Enum):
    REFETCH_ALL_DATASETS = "refetch_all_datasets"
    REFETCH_COMPONENT = "refetch_component"


class AppConstants:
    DB_DATASETS = append_app_data_path(DB_DATASETS)


class NullFillStrategy(Enum):
    NONE = 1
    ZERO = 2
    MEAN = 3
    CLOSEST = 4


class ScalingStrategy(Enum):
    NONE = 1
    MIN_MAX = 2
    STANDARD = 3


class Signals:
    OPEN_TRAINING_TOOLBAR = "SIGNAL_OPEN_TRAINING_TOOLBAR"
    CLOSE_TOOLBAR = "SIGNAL_CLOSE_TOOLBAR"
    EPOCH_COMPLETE = "SIGNAL_EPOCH_COMPLETE\n{EPOCHS_RAN}/{MAX_EPOCHS}/{TRAIN_LOSS}/{VAL_LOSS}/{EPOCH_TIME}/{TRAIN_JOB_ID}"


class CandleSize:
    ONE_MINUTE = "1m"
    THREE_MINUTES = "3m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    SIX_HOURS = "6h"
    EIGHT_HOURS = "8h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    THREE_DAYS = "3d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


ONE_YEAR_IN_MS = 31556952000
ONE_HOUR_IN_MS = 3_600_000
