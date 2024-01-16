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
from tests.t_env import is_fast_test_mode
from tests.t_populate import t_upload_dataset
from tests.t_utils import read_csv_to_df, t_generate_big_dataframe

sys.path.append(SERVER_SOURCE_DIR)

import server
from utils import rm_file, add_to_datasets_db
from db import DatasetUtils, exec_sql
from config import append_app_data_path
from constants import LOG_FILE, DB_DATASETS, BINANCE_DATA_COLS
from sql_statements import CREATE_DATASET_UTILS_TABLE


def t_binance_path_to_dataset_name(binance_path: str):
    parts = binance_path.split("-")
    return parts[0].lower() + "_" + parts[1]


def t_init_server():
    os.environ["APP_DATA_PATH"] = Constants.TESTS_FOLDER
    server.run()


def t_rm_db():
    rm_file(append_app_data_path(DatasetUtils.DB_PATH))
    rm_file(append_app_data_path(LOG_FILE))
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
def fixt_btc_small_1h():
    dataset = BinanceData.BTCUSDT_1H_2023_06
    df = read_csv_to_df(dataset.path)
    df.columns = BINANCE_DATA_COLS
    add_to_datasets_db(df, dataset.name)
    DatasetUtils.create_db_utils_entry(dataset.name, dataset.timeseries_col)
    return dataset


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
    t_rm_db()
    process = multiprocessing.Process(target=t_init_server)
    process.start()
    time.sleep(3)
    yield
    process.terminate()
    process.join()
    t_rm_db()
    rm_file(append_app_data_path(Constants.BIG_TEST_FILE))
