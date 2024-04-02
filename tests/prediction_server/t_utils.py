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
    def create_cloud_log(api_key: str, body):
        with Req("post", URL.create_log(), api_key, json=body) as res:
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
