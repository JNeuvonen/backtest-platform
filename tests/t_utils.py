import requests

from tests.t_constants import URL


def t_fetch_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def t_get_timeseries_col(table_item):
    return table_item["timeseries_col"]


def t_generate_big_dataset(data):
    return


class FetchData:
    @staticmethod
    def get_tables():
        return t_fetch_json(URL.t_get_tables())["tables"]
