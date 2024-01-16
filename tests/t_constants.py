from tests.t_utils import fetch_json


class Constants:
    TESTS_FOLDER = "/tests"


class BinanceData:
    BTCUSDT_1H_2023_06 = "BTCUSDT-1h-2023-06.csv"
    DOGEUSDT_1H_2023_06 = "DOGEUSDT-1h-2023-06.csv"
    ETHBTC_1H_2023_06 = "ETHBTC-1h-2023-06.csv"
    ETHUSDT_1H_2023_06 = "ETHUSDT-1h-2023-06.csv"


class URL:
    BASE_URL = "http://localhost:8000"

    class Datasets:
        ROUTE = "/dataset"
        TABLES = "/tables"

        @staticmethod
        def get_tables_url():
            return URL.BASE_URL + URL.Datasets.ROUTE + URL.Datasets.TABLES


class FetchData:
    @staticmethod
    def get_tables():
        return fetch_json(URL.Datasets.get_tables_url())


BINANCE_FIXTURES_PATH = "fixtures/binance/{}"
