from backtest_utils import MLBasedBacktest
from request_types import BodyMLBasedBacktest


async def run_ml_based_backtest(body: BodyMLBasedBacktest):
    backtest = MLBasedBacktest(body)
    pass
