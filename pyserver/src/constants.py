BINANCE_DATA_COLS = [
    "kline_open_time",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "kline_close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

LOG_FILE = "logs"

DB_DATASETS = "datasets.db"
DB_WORKER_QUEUE = "binance_scrape_worker_queue.db"
DB_LOGS = "logs.db"

BINANCE_TEMP_SCRAPE_PATH = "binance_scraped_data"
BINANCE_UNZIPPED_TEMP_PATH = "binance_unzip_data"
