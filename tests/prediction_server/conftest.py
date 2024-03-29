import pytest
import os
import sys
import subprocess
import multiprocessing
import time

from pandas.compat import platform
from dotenv import load_dotenv

from conf import TEST_RUN_PORT

load_dotenv()


SERVICE_CODE_SOURCE_DIR = "realtime/prediction_server/src"

sys.path.append(SERVICE_CODE_SOURCE_DIR)

from main import start_server
import orm


def start_service():
    os.environ[
        "DATABASE_URI"
    ] = "postgresql://username:password@localhost/live_env_local_test"
    os.environ["SERVICE_PORT"] = TEST_RUN_PORT
    os.environ["ENV"] = "DEV"
    start_server()


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
    orm.drop_tables()
    kill_process_on_port(TEST_RUN_PORT)
    process = multiprocessing.Process(target=start_service)
    process.start()
    time.sleep(5)
    yield
    process.terminate()
    process.join()
    orm.drop_tables()
