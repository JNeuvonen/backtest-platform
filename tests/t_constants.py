import sys
from tests.t_utils import fetch_json

sys.path.append("pyserver/src")
from server import Routers
from route_datasets import RoutePaths


class Constants:
    TESTS_FOLDER = "/tests"


class MockDataset:
    def __init__(self, path: str, dataset_name: str) -> None:
        self.path = path
        self.name = dataset_name


class BinanceData:
    BTCUSDT_1H_2023_06 = MockDataset("BTCUSDT-1h-2023-06.csv", "btcusdt_1h")
    DOGEUSDT_1H_2023_06 = MockDataset("DOGEUSDT-1h-2023-06.csv", "dogeusdt_1h")
    ETHBTC_1H_2023_06 = MockDataset("ETHBTC-1h-2023-06.csv", "ethbtc_1h")
    ETHUSDT_1H_2023_06 = MockDataset("ETHUSDT-1h-2023-06.csv", "ethusdt_1h")


class FixturePaths:
    BINANCE = "fixtures/binance/{}"


class URL:
    BASE_URL = "http://localhost:8000"

    @classmethod
    def _datasets_route(cls):
        return cls.BASE_URL + Routers.DATASET

    @classmethod
    def get_tables(cls):
        return cls._datasets_route() + RoutePaths.ALL_TABLES

    @classmethod
    def get_upload_dataset_url(cls, table_name: str):
        return (
            cls._datasets_route()
            + RoutePaths.UPLOAD_TIMESERIES_DATA
            + f"?dataset_name={table_name}"
        )


class FetchData:
    @staticmethod
    def get_tables():
        return fetch_json(URL.get_tables())
