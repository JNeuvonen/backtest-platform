from backtest_utils import MLBasedBacktest
from query_dataset import DatasetQuery
from request_types import BodyMLBasedBacktest


async def run_ml_based_backtest(body: BodyMLBasedBacktest):
    dataset = DatasetQuery.fetch_dataset_by_name(body.dataset_name)
    backtest = MLBasedBacktest(body, dataset)
    pass
