import requests

from contextlib import contextmanager
from t_constants import URL


@contextmanager
def Req(method, url, api_key, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["X-API-KEY"] = api_key
    with requests.request(method, url, headers=headers, **kwargs) as response:
        response.raise_for_status()
        yield response


class Post:
    @staticmethod
    def create_strategy(api_key: str, body):
        with Req("post", URL.create_strategy(), api_key, json=body) as res:
            return res

    @staticmethod
    def create_strategy_group(api_key: str, body):
        with Req("post", URL.create_strategy_group(), api_key, json=body) as res:
            return res

    @staticmethod
    def create_long_short_strategy(api_key: str, body):
        with Req("post", URL.create_longshort_strategy(), api_key, json=body) as res:
            return res

    @staticmethod
    def create_cloud_log(api_key: str, body):
        with Req("post", URL.create_log(), api_key, json=body) as res:
            return res

    @staticmethod
    def create_account(api_key: str, body):
        with Req("post", URL.create_account(), api_key, json=body) as res:
            return res

    @staticmethod
    def create_trade(api_key: str, body):
        with Req("post", URL.create_trade(), api_key, json=body) as res:
            return res

    @staticmethod
    def enter_longshort_trade(api_key: str, id, body):
        with Req("post", URL.create_longshort_trade(id), api_key, json=body) as res:
            return res

    @staticmethod
    def exit_longshort_trade(api_key: str, id, body):
        with Req("post", URL.exit_longshort_trade(id), api_key, json=body) as res:
            return res


class Get:
    @staticmethod
    def fetch_strategies(api_key: str):
        with Req("get", URL.fetch_strategies(), api_key) as res:
            res_json = res.json()
            return res_json["data"]

    @staticmethod
    def fetch_cloud_logs(api_key: str):
        with Req("get", URL.fetch_logs(), api_key) as res:
            res_json = res.json()
            return res_json["data"]

    @staticmethod
    def fetch_accounts(api_key: str):
        with Req("get", URL.fetch_accounts(), api_key) as res:
            res_json = res.json()
            return res_json["data"]

    @staticmethod
    def fetch_account_by_name(api_key: str, name: str):
        with Req("get", URL.fetch_account_by_name(name), api_key) as res:
            res_json = res.json()
            return res_json["data"]

    @staticmethod
    def fetch_trades(api_key: str):
        with Req("get", URL.fetch_trades(), api_key) as res:
            res_json = res.json()
            return res_json["data"]


class Put:
    @staticmethod
    def update_trade_close(api_key: str, strat_id: int, body):
        with Req("put", URL.update_trade_close(strat_id), api_key, json=body) as res:
            return res

    @staticmethod
    def update_strategy(api_key: str, body):
        with Req("put", URL.update_strategy(), api_key, json=body) as res:
            return res
