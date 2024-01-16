import sys
from contextlib import contextmanager
from tests.t_constants import BINANCE_FIXTURES_PATH

sys.path.append("pyserver/src")

from config import append_app_data_path


@contextmanager
def binance_file(filename):
    try:
        file = open(append_app_data_path(BINANCE_FIXTURES_PATH.format(filename)), "rb")
        yield file
    finally:
        file.close()
