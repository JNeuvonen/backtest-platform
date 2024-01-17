import os
import subprocess
import multiprocessing
from pandas.compat import platform
import pytest
import time
import sys
from tests.t_conf import SERVER_SOURCE_DIR

from tests.t_constants import (
    BinanceCols,
    BinanceData,
    Constants,
    DatasetMetadata,
    Size,
)
from tests.t_download_data import download_historical_binance_data
from tests.t_env import is_fast_test_mode
from tests.t_utils import (
    t_add_binance_dataset_to_db,
    t_generate_big_dataframe,
)

sys.path.append(SERVER_SOURCE_DIR)

import server
from utils import rm_file
from db import DatasetUtils, exec_sql
from config import append_app_data_path
from constants import DB_DATASETS
from sql_statements import CREATE_DATASET_UTILS_TABLE


def t_binance_path_to_dataset_name(binance_path: str):
    parts = binance_path.split("-")
    return parts[0].lower() + "_" + parts[1]


def t_init_server():
    os.environ["APP_DATA_PATH"] = Constants.TESTS_FOLDER
    server.run()


def download_data():
    download_historical_binance_data(
        BinanceData.BTCUSDT_1MO.pair_name,
        "1M",
        append_app_data_path(BinanceData.BTCUSDT_1MO.path),
    )

    download_historical_binance_data(
        BinanceData.SUSHIUSDT_1MO.pair_name,
        "1M",
        append_app_data_path(BinanceData.SUSHIUSDT_1MO.path),
    )
    download_historical_binance_data(
        BinanceData.AAVEUSDT_1MO.pair_name,
        "1M",
        append_app_data_path(BinanceData.AAVEUSDT_1MO.path),
    )


def t_rm_db():
    rm_file(append_app_data_path(DatasetUtils.DB_PATH))
    rm_file(append_app_data_path(DB_DATASETS))


@pytest.fixture
def cleanup_db():
    t_rm_db()
    exec_sql(DatasetUtils.get_path(), CREATE_DATASET_UTILS_TABLE)
    yield


@pytest.fixture
def fixt_init_large_csv():
    if os.path.exists(append_app_data_path(Constants.BIG_TEST_FILE)):
        return
    dataset = BinanceData.BTCUSDT_1H_2023_06
    if is_fast_test_mode():
        df = t_generate_big_dataframe(dataset, Size.MB_BYTES)
    else:
        df = t_generate_big_dataframe(dataset, Size.GB_BYTES)

    df.to_csv(append_app_data_path(Constants.BIG_TEST_FILE), index=False)
    return DatasetMetadata(
        Constants.BIG_TEST_FILE, "btc_big_test_file", BinanceCols.KLINE_OPEN_TIME
    )


@pytest.fixture
def fixt_add_all_downloaded_datasets():
    binance_datasets = [
        BinanceData.BTCUSDT_1MO,
        BinanceData.AAVEUSDT_1MO,
        BinanceData.SUSHIUSDT_1MO,
    ]

    for dataset in binance_datasets:
        t_add_binance_dataset_to_db(dataset)

    return binance_datasets


@pytest.fixture
def fixt_btc_small_1h():
    dataset = BinanceData.BTCUSDT_1H_2023_06
    t_add_binance_dataset_to_db(dataset)
    return dataset


@pytest.fixture
def fixt_add_many_datasets():
    binance_datasets = [
        BinanceData.BTCUSDT_1H_2023_06,
        BinanceData.DOGEUSDT_1H_2023_06,
        BinanceData.ETHUSDT_1H_2023_06,
    ]

    for dataset in binance_datasets:
        t_add_binance_dataset_to_db(dataset)

    return binance_datasets


def t_kill_process_on_port(port):
    try:
        if platform.system() == "Windows":
            command = f"netstat -ano | findstr :{port}"
            lines = subprocess.check_output(command, shell=True).decode().splitlines()
            for line in lines:
                pid = line.strip().split()[-1]
                subprocess.call(f"taskkill /F /PID {pid}", shell=True)
        elif platform.system() == "Darwin":
            command = f"lsof -i tcp:{port} | grep LISTEN"
            lines = subprocess.check_output(command, shell=True).decode().splitlines()
            for line in lines:
                pid = line.split()[1]
                os.kill(int(pid), 9)
        else:
            command = f"fuser -k {port}/tcp"
            subprocess.call(command, shell=True)
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    t_kill_process_on_port(8000)
    download_data()
    t_rm_db()
    process = multiprocessing.Process(target=t_init_server)
    process.start()
    time.sleep(3)
    yield
    process.terminate()
    process.join()
    t_rm_db()
    rm_file(append_app_data_path(Constants.BIG_TEST_FILE))
