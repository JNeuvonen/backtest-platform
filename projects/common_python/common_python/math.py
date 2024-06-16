from common_python.constants import YEAR_IN_MS


def safe_divide(num, denom, fallback=None):
    if denom == 0.0 or denom == 0:
        return fallback
    return num / denom


def get_cagr(end_balance, start_balance, years):
    return (end_balance / start_balance) ** (1 / years) - 1


def ms_to_years(ms):
    return ms / YEAR_IN_MS
