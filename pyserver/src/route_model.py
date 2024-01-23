from fastapi import APIRouter

from context import HttpResponseContext
from code_gen import CodeGen
from orm import ModelQuery, TrainJobQuery
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
        train_job_name = TrainJobQuery.create_train_job(model_name, body)

        train_job = TrainJobQuery.get_train_job(train_job_name, "name")
        model = ModelQuery.fetch_model_by_name(model_name)

        if train_job is None or model is None:
            # should never execute, just to make mypy happy
            raise ValueError("No model or train job found")

        code_gen = CodeGen()
        code_gen.train_job(model, train_job)
        return {"message": "OK"}


@router.get(RoutePaths.TRAIN_JOB_BY_ID)
async def route_fetch_train_job(id: int):
    with HttpResponseContext():
        train = TrainJobQuery.get_train_job(id)
        return {"train": train}
