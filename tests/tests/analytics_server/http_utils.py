import requests
from contextlib import contextmanager
from common_python.test_utils.conf import TEST_RUN_PORT
from analytics_server.api.v1.user import RoutePaths as UserRoutePaths
from analytics_server.main import Routers
from tests.analytics_server.fixtures.test_token import TEST_TOKEN


@contextmanager
def Req(method, url, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["authorization"] = f"Bearer {TEST_TOKEN}"
    with requests.request(method, url, headers=headers, **kwargs) as response:
        response.raise_for_status()
        yield response


class URL:
    BASE_URL = f"http://localhost:{TEST_RUN_PORT}"

    @classmethod
    def _user_route(cls):
        return cls.BASE_URL + Routers.V1_USER

    @staticmethod
    def create_user():
        return URL._user_route() + UserRoutePaths.ROOT

    @staticmethod
    def get_user_root():
        return URL._user_route() + UserRoutePaths.ROOT


class Post:
    @staticmethod
    def create_user(body):
        with Req("post", URL.create_user(), json=body) as res:
            return res


class Get:
    @staticmethod
    def call_user_root():
        with Req("get", URL.get_user_root()) as res:
            return res
