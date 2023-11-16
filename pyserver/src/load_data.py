from sqlite3 import Connection
import time
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import requests
import os
import zipfile
import shutil
from datasets import combine
from constants import DATASETS_DB
from db import create_connection


class ScrapeJob:
    def __init__(self, url, target_dataset=False, columns_to_drop=[]):
        self.url = url
        self.margin_type = None
        self.prefix = None
        self.columns_to_drop = columns_to_drop
        self.target_dataset = target_dataset
        self.path = self.process_url()

    def is_spot_url(self):
        return "spot" in self.url

    def get_prefix(self):
        return self.pair + "_" + self.market_type + "_" + self.candle_interval + "_"

    def process_url(self):
        if self.is_spot_url():
            url_parts = self.url.split("/")
            self.dataseries_interval = url_parts[5]
            self.klines = url_parts[6]
            self.pair = url_parts[7]
            self.candle_interval = url_parts[8]
            self.market_type = "spot"
            return os.path.join(
                "..",
                "..",
                "..",
                "data",
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
                "..",
                "..",
                "..",
                "data",
                self.market_type,
                self.margin_type,
                self.klines,
                self.pair,
                self.candle_interval,
            )


def is_spot_url(url):
    return "spot" in url


def endpoint_type(url):
    return "metrics" in url


def get_datatick_interval(scrape_type, fallback):
    if scrape_type == "metrics":
        return "5m"

    return fallback


def load_data(job: ScrapeJob, conn: Connection):
    service = Service()
    driver = webdriver.Chrome(service=service)
    driver.get(job.url)
    time.sleep(2)
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
            download_file(href)

    driver.quit()

    write_to_disk("scraped_data", job, conn)


def load_all_datasets(jobs: List[ScrapeJob], conn: (Connection | None)):
    if conn is None:
        return

    for item in jobs:
        load_data(item, conn)


def write_to_disk(input_dir, scrape_object: ScrapeJob, conn: Connection):
    if not os.path.exists(scrape_object.path):
        os.makedirs(scrape_object.path)

    files = os.listdir(input_dir)

    for file in files:
        if file.endswith(".zip"):
            file_path = os.path.join(input_dir, file)

            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(scrape_object.path)
    combine(
        conn,
        scrape_object.path,
        scrape_object.pair,
        scrape_object.market_type,
        scrape_object.dataseries_interval,
        scrape_object.candle_interval,
    )
    shutil.rmtree(input_dir)


def download_file(url):
    directory = "scraped_data"
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


if __name__ == "__main__":
    db = create_connection(DATASETS_DB)

    jobs = [
        ScrapeJob(
            "https://data.binance.vision/?prefix=data/spot/monthly/klines/IOTAUSDT/1h/",
            True,
            [],
        ),
    ]
    load_all_datasets(jobs, db)
