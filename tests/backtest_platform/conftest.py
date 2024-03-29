import os
import subprocess
import multiprocessing
from pandas.compat import platform
import pytest
import time
import sys
from tests.backtest_platform.fixtures import (
    close_long_trade_cond_basic,
    close_short_trade_cond_basic,
    create_code_preset_body,
    create_manual_backtest,
    create_train_job_basic,
    criterion_basic,
    linear_model_basic,
    open_long_trade_cond_basic,
    open_short_trade_cond_basic,
)
from tests.backtest_platform.t_conf import SERVER_SOURCE_DIR

from tests.backtest_platform.t_constants import (
    BinanceCols,
    BinanceData,
    CodePresetId,
    Constants,
    DatasetMetadata,
    Size,
)
from tests.backtest_platform.t_download_data import download_historical_binance_data
from tests.backtest_platform.t_env import is_fast_test_mode
from tests.backtest_platform.t_utils import (
    Fetch,
    Post,
    create_model_body,
    t_add_binance_dataset_to_db,
    t_generate_big_dataframe,
)

sys.path.append(SERVER_SOURCE_DIR)

import server
from utils import rm_file
from config import append_app_data_path
from constants import DB_DATASETS, NullFillStrategy, DATASET_UTILS_DB_PATH
from orm import create_tables, db_delete_all_data
from query_trainjob import TrainJobQuery


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


@pytest.fixture
def cleanup_db():
    db_delete_all_data()
    yield
    db_delete_all_data()


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
        Constants.BIG_TEST_FILE,
        "btc_big_test_file",
        BinanceCols.KLINE_OPEN_TIME,
        BinanceCols.OPEN_PRICE,
        BinanceCols.OPEN_PRICE,
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
def fixt_manual_backtest(fixt_btc_small_1h):
    dataset = Fetch.get_dataset_by_name(fixt_btc_small_1h.name)
    backtest_body = create_manual_backtest(
        dataset["id"],
        True,
        open_long_trade_cond_basic(),
        open_short_trade_cond_basic(),
        close_long_trade_cond_basic(),
        close_short_trade_cond_basic(),
        False,
        0.1,
        0.01,
        [0, 100],
        False,
        False,
        0.000016,
        0.0,
        0.0,
    )
    Post.create_manual_backtest(backtest_body)
    return fixt_btc_small_1h


@pytest.fixture
def fixt_create_code_preset():
    code_preset = create_code_preset_body(
        code="Hello world",
        category=CodePresetId.CREATE_COLUMNS,
        name="code_preset_name",
    )
    id = Post.create_code_preset(code_preset)
    return id


@pytest.fixture
def create_basic_model(fixt_btc_small_1h):
    """
    Creates a basic model using the 'fixt_btc_small_1h' fixture.
    The 'fixt_btc_small_1h' fixture is executed as a dependency.
    """
    body = create_model_body(
        name=Constants.EXAMPLE_MODEL_NAME,
        drop_cols=[],
        null_fill_strategy=NullFillStrategy.CLOSEST.value,
        model=linear_model_basic(),
        hyper_params_and_optimizer_code=criterion_basic(),
        validation_split=[70, 100],
    )
    Post.create_model(fixt_btc_small_1h.name, body)
    return fixt_btc_small_1h, body


@pytest.fixture
def create_train_job(create_basic_model):
    train_job_id = Post.create_train_job(
        Constants.EXAMPLE_MODEL_NAME, body=create_train_job_basic()
    )
    train_job = TrainJobQuery.get_train_job(train_job_id)
    return create_basic_model[0], create_basic_model[1], train_job


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


def del_db_files():
    rm_file(append_app_data_path(Constants.BIG_TEST_FILE))
    rm_file(append_app_data_path(DATASET_UTILS_DB_PATH))
    rm_file(append_app_data_path(DB_DATASETS))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    del_db_files()
    create_tables()
    t_kill_process_on_port(8000)
    download_data()
    process = multiprocessing.Process(target=t_init_server)
    process.start()
    time.sleep(5)
    yield
    process.terminate()
    process.join()
    del_db_files()
