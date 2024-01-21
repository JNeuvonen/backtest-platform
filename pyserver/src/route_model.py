from fastapi import APIRouter

from context import HttpResponseContext
from db import DatasetUtils


router = APIRouter()


class RoutePaths:
    FETCH_MODEL = "/{model_name}"


@router.get(RoutePaths.FETCH_MODEL)
async def route_fetch_model(model_name: str):
    with HttpResponseContext():
        model = DatasetUtils.fetch_model_by_name(model_name)
        return {"model": model}
