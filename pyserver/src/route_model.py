from fastapi import APIRouter

from context import HttpResponseContext
from db import DatasetUtils
from pyserver.src.request_types import BodyCreateTrain


router = APIRouter()


class RoutePaths:
    FETCH_MODEL = "/{model_name}"
    CREATE_TRAIN_JOB = "/{model_name}/create-train"
    TRAIN_JOB_BY_ID = "/train/{id}"


@router.get(RoutePaths.FETCH_MODEL)
async def route_fetch_model(model_name: str):
    with HttpResponseContext():
        model = DatasetUtils.fetch_model_by_name(model_name)
        return {"model": model}


@router.post(RoutePaths.CREATE_TRAIN_JOB)
async def route_create_train_job(model_name: str, body: BodyCreateTrain):
    with HttpResponseContext():
        train_job_name = DatasetUtils.create_train_job(model_name, body)
        train_job = DatasetUtils.get_train_job(
            train_job_name, DatasetUtils.TrainJob.Cols.NAME
        )
        return {"message": "OK"}


@router.get(RoutePaths.TRAIN_JOB_BY_ID)
async def route_fetch_train_job(id: int):
    with HttpResponseContext():
        train = DatasetUtils.get_train_job(id)
        return {"train": train}
