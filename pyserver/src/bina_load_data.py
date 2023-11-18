from sqlite3 import Connection
import time
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import os
import zipfile
import shutil
from constants import BINANCE_TEMP_SCRAPE_PATH, BINANCE_UNZIPPED_TEMP_PATH
from datasets import combine
from db_models import RowScrapeJob
from db_statements import DELETE_SCRAPE_JOB
from log import Logger


def install_web_driver(app_data_path):
    os.environ["WDM_LOCAL"] = "1"
    os.environ["WDM_CACHE_DIR"] = app_data_path

    options = Options()
    options.add_argument("--headless")

    service = Service(ChromeDriverManager().install())

    return webdriver.Chrome(service=service, options=options)


class ScrapeJob:
    def __init__(self, url, target_dataset=False, columns_to_drop=[]):
        self.url = url
        self.margin_type = None
        self.prefix = None
        self.columns_to_drop = columns_to_drop
        self.target_dataset = target_dataset
        self.path = self.process_url()
        self.invalid_path = False

    def is_spot_url(self):
        return "spot" in self.url

    def get_prefix(self):
        return self.pair + "_" + self.market_type + "_" + self.candle_interval + "_"

    def process_url(self):
        try:
            if self.is_spot_url():
                url_parts = self.url.split("/")
                self.dataseries_interval = url_parts[5]
                self.klines = url_parts[6]
                self.pair = url_parts[7]
                self.candle_interval = url_parts[8]
                self.market_type = "spot"
                return os.path.join(
                    BINANCE_UNZIPPED_TEMP_PATH,
                    self.market_type,
                    self.klines,
                    self.pair,
                    self.candle_interval,
                )
            else:
                url_parts = self.url.split("/")
                prefix_index = url_parts.index("?prefix=data") + 1
                self.margin_type = url_parts[prefix_index + 1]
                self.dataseries_interval = url_parts[prefix_index + 2]
                self.klines = url_parts[prefix_index + 3]
                self.pair = url_parts[prefix_index + 4]
                self.candle_interval = url_parts[prefix_index + 5]
                type_of_endpoint = endpoint_type(self.url)
                self.candle_interval = get_datatick_interval(
                    type_of_endpoint, self.candle_interval
                )
                self.market_type = "futures"
                return os.path.join(
                    BINANCE_UNZIPPED_TEMP_PATH,
                    self.market_type,
                    self.margin_type,
                    self.klines,
                    self.pair,
                    self.candle_interval,
                )
        except (ValueError, IndexError):
            self.invalid_path = True
            return None


def is_spot_url(url):
    return "spot" in url


def endpoint_type(url):
    return "metrics" in url


def get_datatick_interval(scrape_type, fallback):
    if scrape_type == "metrics":
        return "5m"

    return fallback


def load_data(
    logger: Logger,
    app_data_path: str,
    job: RowScrapeJob,
    db_datasets_conn: Connection,
    db_worker_queue_conn: Connection,
):
    try:
        logger.info(f"Initiating logging on {job.url}")
        driver = install_web_driver(app_data_path)
        driver.get(job.url)
        time.sleep(2)
        logger.info(f"Launched selenium bot on: {job.url}")
        parsed_page = BeautifulSoup(driver.page_source, "html.parser")

        trs = parsed_page.find_all("tr")
        count = 0
        for tr in trs:
            count += 1
            if count < 3:
                continue
            td = tr.find("td")
            anchor = td.find("a")
            href = anchor["href"]

            if href.endswith(".zip") and "CHECKSUM" not in href:
                download_file(app_data_path, href)

        driver.quit()

        write_to_disk(
            app_data_path,
            job,
            db_datasets_conn,
            db_worker_queue_conn,
        )
    except Exception as e:
        logger.error(f"Exception running scrape on: {job.url}: {e}")


def write_to_disk(
    app_data_path: str,
    scrape_object: RowScrapeJob,
    db_datasets_conn: Connection,
    db_worker_queue_conn: Connection,
):
    if scrape_object.path is None:
        return

    if not os.path.exists(os.path.join(app_data_path, scrape_object.path)):
        os.makedirs(os.path.join(app_data_path, scrape_object.path))

    files = os.listdir(os.path.join(app_data_path, BINANCE_TEMP_SCRAPE_PATH))

    for file in files:
        if file.endswith(".zip"):
            file_path = os.path.join(app_data_path, BINANCE_TEMP_SCRAPE_PATH, file)

            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(app_data_path, scrape_object.path))
    combine(
        db_datasets_conn,
        os.path.join(app_data_path, scrape_object.path),
        scrape_object.pair,
        scrape_object.market_type,
        scrape_object.dataseries_interval,
        scrape_object.candle_interval,
    )
    db_worker_cursor = db_worker_queue_conn.cursor()
    db_worker_cursor.execute(DELETE_SCRAPE_JOB, (scrape_object.id,))
    db_worker_queue_conn.commit()
    db_worker_cursor.close()
    shutil.rmtree(os.path.join(app_data_path, BINANCE_TEMP_SCRAPE_PATH))
    shutil.rmtree(os.path.join(app_data_path, BINANCE_UNZIPPED_TEMP_PATH))


def download_file(app_data_path: str, url: str):
    directory = os.path.join(app_data_path, BINANCE_TEMP_SCRAPE_PATH)

    if not os.path.exists(directory):
        os.makedirs(directory)

    local_filename = url.split("/")[-1]
    local_path = os.path.join(directory, local_filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path
