import os
import time
import threading


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
