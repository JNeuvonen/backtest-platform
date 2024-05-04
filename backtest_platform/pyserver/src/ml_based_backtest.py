import json
from api_binance import non_async_save_historical_klines
from backtest_utils import (
    MLBasedBacktest,
    exec_load_model,
    run_transformations_on_dataset,
)
from constants import BINANCE_BACKTEST_PRICE_COL
from dataset import read_dataset_to_mem
from query_data_transformation import DataTransformationQuery
from query_dataset import DatasetQuery
from query_model import ModelQuery
from query_trainjob import TrainJobQuery
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

    model_metadata = ModelQuery.fetch_model_by_id(body.id_of_model)
    trainjob = TrainJobQuery.get_train_job(body.train_run_id)
    drop_cols = []

    model = exec_load_model(
        model_class_code=model_metadata.model_code,
        train_job_id=trainjob.id,
        epoch_nr=body.epoch,
        x_input_size=model_metadata.x_num_cols,
    )

    backtest = MLBasedBacktest(body, dataset)
