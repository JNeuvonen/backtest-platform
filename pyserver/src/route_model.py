from fastapi import APIRouter

from context import HttpResponseContext
from db import DatasetUtils
from pyserver.src.request_types import BodyCreateTrain


router = APIRouter()


class RoutePaths:
    FETCH_MODEL = "/{model_name}"
    CREATE_TRAIN_JOB = "/{model_name}/create-train"


@router.get(RoutePaths.FETCH_MODEL)
async def route_fetch_model(model_name: str):
    with HttpResponseContext():
        model = DatasetUtils.fetch_model_by_name(model_name)
        return {"model": model}


@router.post(RoutePaths.CREATE_TRAIN_JOB)
async def route_create_train_job(model_name: str, body: BodyCreateTrain):
    with HttpResponseContext():
        DatasetUtils.create_train_job(model_name, body)
        return {"message": "OK"}
