import logging
from typing import List
from api_binance import non_async_save_historical_klines
from constants import DomEventChannels
from db import exec_python
from log import get_logger
from query_data_transformation import DataTransformationQuery

from query_dataset import DatasetQuery
from request_types import BodyCloneIndicators
from utils import PythonCode, get_binance_dataset_tablename


def clone_indicators_to_existing_datasets(
    dataset_name, existing_datasets: List[str], notify_frontend=False
):
    if len(existing_datasets) == 0:
        return

    logger = get_logger()
    base_dataset = DatasetQuery.fetch_dataset_by_name(dataset_name)

    data_transformations = DataTransformationQuery.get_transformations_by_dataset(
        base_dataset.id
    )

    for dataset_name in existing_datasets:
        dataset = DatasetQuery.fetch_dataset_by_name(dataset_name)
        for transformation in data_transformations:
            transformation_code = transformation.transformation_code
            python_program = PythonCode.on_dataset(dataset_name, transformation_code)
            exec_python(python_program)
            DataTransformationQuery.create_entry(
                {
                    "dataset_id": dataset.id,
                    "transformation_code": transformation_code,
                }
            )

        DatasetQuery.update_target_column(
            dataset.dataset_name, base_dataset.target_column
        )
        DatasetQuery.update_price_column(
            dataset.dataset_name, base_dataset.price_column
        )

    if notify_frontend is True:
        logger.log(
            "Finished cloning indicators to existing datasets",
            logging.INFO,
            True,
            True,
            DomEventChannels.REFETCH_ALL_DATASETS.value,
        )


def clone_indicators_to_new_datasets(
    dataset_name, body: BodyCloneIndicators, notify_frontend=False
):
    if len(body.new_datasets) == 0:
        return

    logger = get_logger()
    base_dataset = DatasetQuery.fetch_dataset_by_name(dataset_name)
    data_transformations = DataTransformationQuery.get_transformations_by_dataset(
        base_dataset.id
    )
    for symbol_name in body.new_datasets:
        if body.candle_interval is None:
            raise Exception("Candle interval is not set")

        if body.use_futures is None:
            raise Exception("use_futures field should be set on request body")

        non_async_save_historical_klines(
            symbol_name, body.candle_interval, True, body.use_futures
        )

        dataset_name = get_binance_dataset_tablename(
            symbol_name, body.candle_interval, body.use_futures
        )
        dataset = DatasetQuery.fetch_dataset_by_name(dataset_name)
        for transformation in data_transformations:
            transformation_code = transformation.transformation_code

            python_program = PythonCode.on_dataset(dataset_name, transformation_code)
            exec_python(python_program)
            DataTransformationQuery.create_entry(
                {
                    "dataset_id": dataset.id,
                    "transformation_code": transformation_code,
                }
            )

        DatasetQuery.update_target_column(dataset_name, base_dataset.target_column)
        DatasetQuery.update_price_column(dataset_name, base_dataset.price_column)

    if notify_frontend is True:
        logger.log(
            "Finished cloning indicators to new datasets",
            logging.INFO,
            True,
            True,
            DomEventChannels.REFETCH_ALL_DATASETS.value,
        )
