import pytest
import subprocess
import os
import multiprocessing
import time
from pandas.compat import platform
from common_python.pred_serv_orm import drop_tables
from common_python.test_utils.conf import DROP_TABLES, TEST_RUN_PORT
from common_python.pred_serv_orm import create_tables, engine
from analytics_server.main import start_server


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


@pytest.fixture
def cleanup_db():
    if DROP_TABLES == 1:
        drop_tables(engine)
    create_tables()
    yield
    if DROP_TABLES == 1:
        drop_tables(engine)
    create_tables()


def start_service():
    start_server()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    if DROP_TABLES is None:
        raise Exception("Provide DROP_TABLES (0 or 1) env variable to the test run.")

    if int(DROP_TABLES) == 1:
        drop_tables(engine)

    create_tables()
    kill_process_on_port(TEST_RUN_PORT)

    server_process = multiprocessing.Process(target=start_service, daemon=False)
    server_process.start()
    time.sleep(5)  # Allow server to start
    yield
    server_process.terminate()
    server_process.join(timeout=5)
    kill_process_on_port(TEST_RUN_PORT)  # Ensure the process is killed
