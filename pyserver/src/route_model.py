import asyncio
from fastapi import APIRouter

from context import HttpResponseContext
from orm import ModelQuery, TrainJobQuery
from code_gen import start_train_loop
from config import is_testing
from request_types import BodyCreateTrain


router = APIRouter()


class RoutePaths:
    FETCH_MODEL = "/{model_name}"
    CREATE_TRAIN_JOB = "/{model_name}/create-train"
    TRAIN_JOB_BY_ID = "/train/{id}"


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
            # should never execute, just to make mypy happy
            raise ValueError("No model or train job found")

        if is_testing() is False:
            asyncio.create_task(start_train_loop(model, train_job))
            return {"id": train_job_id}

        ##blocking in test mode
        await start_train_loop(model, train_job)
        return {"id": train_job_id}


@router.get(RoutePaths.TRAIN_JOB_BY_ID)
async def route_fetch_train_job(id: int):
    with HttpResponseContext():
        train = TrainJobQuery.get_train_job(id)
        return {"train": train}
