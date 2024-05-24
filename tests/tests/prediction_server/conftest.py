import pytest
import os
import subprocess
import multiprocessing
import time
import secrets

from pandas.compat import platform
from dotenv import load_dotenv

from conf import TEST_RUN_PORT, DROP_TABLES
from import_helper import (
    start_server,
    drop_tables,
    stop_server,
    create_api_key_entry,
    create_tables,
)
from fixtures.account import create_master_acc
from t_utils import Post

load_dotenv()


def start_service():
    start_server()


def generate_api_key_for_testrun():
    pass


@pytest.fixture
def fixt_create_master_acc(create_api_key):
    master_acc = create_master_acc()
    Post.create_account(create_api_key, master_acc)
    return create_api_key


@pytest.fixture
def create_api_key():
    api_key = secrets.token_urlsafe(32)
    create_api_key_entry(api_key)
    return api_key


@pytest.fixture
def cleanup_db():
    if DROP_TABLES == 1:
        drop_tables()
    create_tables()
    yield
    if DROP_TABLES == 1:
        drop_tables()
    create_tables()


def kill_process_on_port(port):
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
    if DROP_TABLES is None:
        raise Exception("Provide DROP_TABLES (0 or 1) env variable to the test run.")

    if int(DROP_TABLES) == 1:
        drop_tables()
    create_tables()
    kill_process_on_port(TEST_RUN_PORT)
    server_process = multiprocessing.Process(target=start_service, daemon=False)
    server_process.start()
    time.sleep(15)  # Allow server to start
    yield
    stop_server()
    server_process.terminate()
    server_process.join(timeout=5)
    kill_process_on_port(TEST_RUN_PORT)  # Ensure the process is killed
    if DROP_TABLES == 1:
        drop_tables()
