import os
import sqlite3
import time
import threading
from typing import List
import pandas as pd

from code_gen_templates import PyCode


NUM_REQ_KLINES_BUFFER = 5


def run_in_thread(fn, *args, **kwargs):
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
    thread.start()
    return thread


def replace_placeholders_on_code_templ(code, replacements):
    for key, value in replacements.items():
        code = code.replace(key, str(value))
    return code


def file_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def get_current_timestamp_ms() -> int:
    return int(time.time() * 1000)


def calculate_timestamp_for_kline_fetch(num_required_klines, kline_size_ms):
    curr_time_ms = get_current_timestamp_ms()

    return curr_time_ms - (
        kline_size_ms * (num_required_klines + NUM_REQ_KLINES_BUFFER)
    )


def read_dataset_to_mem(db_path: str, table_name: str):
    with sqlite3.connect(db_path) as conn:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        return df


def gen_data_transformations_code(data_transformations: List):
    sorted_transformations = sorted(data_transformations, key=lambda x: x.strategy_id)

    data_transformations_code = PyCode()
    data_transformations_code.append_line("def make_data_transformations(dataset):")
    data_transformations_code.add_indent()
    for item in sorted_transformations:
        data_transformations_code.add_block(item.transformation_code)

    data_transformations_code.append_line("return dataset")
    return data_transformations_code.get()
