import asyncio
from fastapi import APIRouter, Response, status
from model_backtest import run_model_backtest

from context import HttpResponseContext
from db import get_dataset_columns
from query_backtest import BacktestQuery
from query_dataset import DatasetQuery
from query_model import ModelQuery
from query_trainjob import TrainJobQuery
from code_gen import start_train_loop
from config import is_testing
from request_types import BodyCreateTrain, BodyRunBacktest


router = APIRouter()


class RoutePaths:
    FETCH_MODEL = "/{model_name}"
    CREATE_TRAIN_JOB = "/{model_name}/create-train"
    TRAIN_JOB_BY_ID = "/train/{id}"
    ALL_METADATA_BY_MODEL_NAME = "/{model_name}/trains"
    STOP_TRAIN = "/train/stop/{train_job_id}"
    TRAIN_JOB_AND_ALL_WEIGHT_METADATA_BY_ID = "/train/{train_job_id}/detailed"
    RUN_BACKTEST = "/backtest/{train_job_id}/run"
    BACKTESTS = "/backtest/{train_job_id}"


@router.get(RoutePaths.FETCH_MODEL)
async def route_fetch_model(model_name: str):
    with HttpResponseContext():
        model = ModelQuery.fetch_model_by_name(model_name)
        return {"model": model}


@router.post(RoutePaths.CREATE_TRAIN_JOB)
async def route_create_train_job(model_name: str, body: BodyCreateTrain):
    with HttpResponseContext():
        train_job_id = TrainJobQuery.create_train_job(model_name, body)

        train_job = TrainJobQuery.get_train_job(train_job_id)
        model = ModelQuery.fetch_model_by_name(model_name)

        if train_job is None or model is None:
            raise ValueError("No model or train job found")

        if is_testing():
            await start_train_loop(model, train_job)
            return {"id": train_job_id}

        asyncio.create_task(start_train_loop(model, train_job))
        return {"id": train_job_id}


@router.get(RoutePaths.TRAIN_JOB_BY_ID)
async def route_fetch_train_job(id: int):
    with HttpResponseContext():
        train = TrainJobQuery.get_train_job(id)
        return {"train": train}


@router.get(RoutePaths.ALL_METADATA_BY_MODEL_NAME)
async def route_fetch_all_metadata_by_name(model_name: str):
    with HttpResponseContext():
        ret = TrainJobQuery.fetch_all_metadata_by_name(model_name)
        return {"data": ret}


@router.post(RoutePaths.STOP_TRAIN)
async def route_stop_train(train_job_id: int):
    with HttpResponseContext():
        TrainJobQuery.set_training_status(train_job_id, False)
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.get(RoutePaths.TRAIN_JOB_AND_ALL_WEIGHT_METADATA_BY_ID)
async def route_fetch_train_job_detailed(train_job_id: int):
    with HttpResponseContext():
        data = TrainJobQuery.get_train_job_detailed(train_job_id)
        dataset_name = data["dataset_metadata"].dataset_name
        cols = get_dataset_columns(dataset_name)
        data["dataset_columns"] = cols
        return {"data": data}


@router.post(RoutePaths.RUN_BACKTEST)
async def route_run_backtest(train_job_id: int, body: BodyRunBacktest):
    with HttpResponseContext():
        price_col = DatasetQuery.get_price_col(body.dataset_name)
        if price_col is None:
            DatasetQuery.update_price_column(body.dataset_name, body.price_col)
        res = run_model_backtest(train_job_id, body)
        return {"data": res}


@router.get(RoutePaths.BACKTESTS)
async def route_get_backtests(train_job_id: int):
    with HttpResponseContext():
        data = BacktestQuery.fetch_backtests_by_train_job_id(train_job_id)
        return {"data": data}
