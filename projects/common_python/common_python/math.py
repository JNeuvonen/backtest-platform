from common_python.constants import YEAR_IN_MS


MINUTE_IN_MS = 60 * 1000
HOUR_IN_MS = 60 * MINUTE_IN_MS
DAY_IN_MS = 24 * HOUR_IN_MS


def safe_divide(num, denom, fallback=None):
    if denom == 0.0 or denom == 0:
        return fallback
    return num / denom


def get_cagr(end_balance, start_balance, years):
    return (end_balance / start_balance) ** (1 / years) - 1


def ms_to_years(ms):
    return ms / YEAR_IN_MS


def get_interval_length_in_ms(interval: str) -> int:
    intervals = {
        "1m": MINUTE_IN_MS,
        "3m": MINUTE_IN_MS * 3,
        "5m": MINUTE_IN_MS * 5,
        "15m": MINUTE_IN_MS * 15,
        "30m": MINUTE_IN_MS * 30,
        "1h": HOUR_IN_MS,
        "2h": HOUR_IN_MS * 2,
        "4h": HOUR_IN_MS * 4,
        "6h": HOUR_IN_MS * 6,
        "8h": HOUR_IN_MS * 8,
        "12h": HOUR_IN_MS * 12,
        "1d": DAY_IN_MS,
        "3d": DAY_IN_MS * 3,
        "1w": DAY_IN_MS * 7,
        "1M": DAY_IN_MS * 30,
    }
    return intervals.get(interval, 0)


def get_klines_required_for_1d(kline_size_ms: int):
    return safe_divide(DAY_IN_MS, kline_size_ms, 0)
