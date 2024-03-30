import requests

from contextlib import contextmanager
from constants import URL


@contextmanager
def Req(method, url, **kwargs):
    with requests.request(method, url, **kwargs) as response:
        response.raise_for_status()
        yield response


class Post:
    @staticmethod
    def create_strategy(body):
        with Req("post", URL.create_strategy(), json=body) as res:
            return res


class Get:
    @staticmethod
    def fetch_strategies():
        with Req("get", URL.fetch_strategies()) as res:
            res_json = res.json()
            return res_json["data"]
