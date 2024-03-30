import pytest
import os
import subprocess
import multiprocessing
import time

from pandas.compat import platform
from dotenv import load_dotenv

from conf import TEST_RUN_PORT
from import_helper import start_server, drop_tables, stop_server, db_delete_all_data

load_dotenv()


def start_service():
    start_server()


@pytest.fixture
def cleanup_db():
    db_delete_all_data()
    yield
    db_delete_all_data()


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
    drop_tables()
    kill_process_on_port(TEST_RUN_PORT)
    server_process = multiprocessing.Process(target=start_service, daemon=True)
    server_process.start()
    time.sleep(5)  # Allow server to start
    yield
    stop_server()
    server_process.terminate()
    server_process.join(timeout=5)
    kill_process_on_port(TEST_RUN_PORT)  # Ensure the process is killed
    drop_tables()
