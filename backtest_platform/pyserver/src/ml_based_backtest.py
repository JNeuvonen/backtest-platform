from api_binance import non_async_save_historical_klines, save_historical_klines
from backtest_utils import MLBasedBacktest, run_transformations_on_dataset
from constants import BINANCE_BACKTEST_PRICE_COL
from dataset import read_dataset_to_mem
from query_data_transformation import DataTransformationQuery
from query_dataset import DatasetQuery
from request_types import BodyMLBasedBacktest


async def run_ml_based_backtest(body: BodyMLBasedBacktest):
    dataset = DatasetQuery.fetch_dataset_by_name(body.dataset_name)
    data_transformations = DataTransformationQuery.get_transformations_by_dataset(
        dataset.id
    )

    if body.fetch_latest_data is True:
        non_async_save_historical_klines(dataset.symbol, dataset.interval, True)
        DatasetQuery.update_price_column(
            dataset.dataset_name, BINANCE_BACKTEST_PRICE_COL
        )
        run_transformations_on_dataset(data_transformations, dataset)

    dataset_df = read_dataset_to_mem(dataset.dataset_name)

    columns = dataset_df.columns.tolist()

    backtest = MLBasedBacktest(body, dataset)
