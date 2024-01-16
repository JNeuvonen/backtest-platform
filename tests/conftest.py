import os
import subprocess
import multiprocessing
import pandas as pd
from pandas.compat import platform
import pytest
import time
import sys
from tests.t_conf import SERVER_SOURCE_DIR

from tests.t_constants import Constants, FixturePaths

sys.path.append(SERVER_SOURCE_DIR)

import server
from utils import rm_file
from db import DatasetUtils, exec_sql
from config import append_app_data_path
from constants import LOG_FILE, DB_DATASETS
from sql_statements import CREATE_DATASET_UTILS_TABLE


def t_binance_path_to_dataset_name(binance_path: str):
    parts = binance_path.split("-")
    return parts[0].lower() + "_" + parts[1]


def t_read_binance_df(dataset_name):
    return pd.read_csv(append_app_data_path(FixturePaths.BINANCE.format(dataset_name)))


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
        print("No process to kill")


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
