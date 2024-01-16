import sys

from tests.t_conf import SERVER_SOURCE_DIR

sys.path.append(SERVER_SOURCE_DIR)

from server import Routers
from route_datasets import RoutePaths


class FixturePaths:
    BINANCE = "fixtures/binance/{}"


class Constants:
    TESTS_FOLDER = "/tests"
    BIG_TEST_FILE = "big_testfile.csv"


class DatasetMetadata:
    def __init__(self, path: str, dataset_name: str, timeseries_col: str) -> None:
        self.path = path
        self.name = dataset_name
        self.timeseries_col = timeseries_col


class BinanceCols:
    KLINE_OPEN_TIME = "kline_open_time"
    OPEN_PRICE = "open_price"
    HIGH_PRICE = "high_price"
    LOW_PRICE = "low_price"
    CLOSE_PRICE = "close_price"
    VOLUME = "volume"
    KLINE_CLOSE_TIME = "kline_close_time"
    QUOTE_ASSET_VOLUME = "quote_asset_volume"
    NUMBER_OF_TRADES = "number_of_trades"
    TAKER_BUY_BASE_ASSET_VOLUME = "taker_buy_base_asset_volume"
    TAKER_BUY_QUOTE_ASSET_VOLUME = "taker_buy_quote_asset_volume"


class BinanceData:
    BTCUSDT_1H_2023_06 = DatasetMetadata(
        FixturePaths.BINANCE.format("BTCUSDT-1h-2023-06.csv"),
        "btcusdt_1h",
        BinanceCols.KLINE_OPEN_TIME,
    )
    DOGEUSDT_1H_2023_06 = DatasetMetadata(
        FixturePaths.BINANCE.format("DOGEUSDT-1h-2023-06.csv"),
        "dogeusdt_1h",
        BinanceCols.KLINE_OPEN_TIME,
    )
    ETHBTC_1H_2023_06 = DatasetMetadata(
        FixturePaths.BINANCE.format("ETHBTC-1h-2023-06.csv"),
        "ethbtc_1h",
        BinanceCols.KLINE_OPEN_TIME,
    )
    ETHUSDT_1H_2023_06 = DatasetMetadata(
        FixturePaths.BINANCE.format("ETHUSDT-1h-2023-06.csv"),
        "ethusdt_1h",
        BinanceCols.KLINE_OPEN_TIME,
    )


class EnvTestSpeed:
    FAST = "FAST"
    SLOW = "SLOW"


class Size:
    KB_BYTES = 1024
    MB_BYTES = 1024 * 1024
    GB_BYTES = 1024 * 1024 * 1024


class URL:
    BASE_URL = "http://localhost:8000"

    @classmethod
    def _datasets_route(cls):
        return cls.BASE_URL + Routers.DATASET

    @classmethod
    def t_get_tables(cls):
        return cls._datasets_route() + RoutePaths.ALL_TABLES

    @classmethod
    def t_get_upload_dataset_url(cls, table_name: str, timeseries_col: str):
        return (
            cls._datasets_route()
            + RoutePaths.UPLOAD_TIMESERIES_DATA
            + f"?dataset_name={table_name}&timeseries_col={timeseries_col}"
        )
