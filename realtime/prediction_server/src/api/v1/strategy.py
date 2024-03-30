from fastapi import APIRouter, Response, status
from context import HttpResponseContext
from api.v1.request_types import BodyCreateStrategy
from schema.strategy import StrategyQuery


router = APIRouter()


class RoutePaths:
    STRATEGY = "/"


@router.get(RoutePaths.STRATEGY)
async def route_get_root():
    with HttpResponseContext():
        strategies = StrategyQuery.get_strategies()
        return {"data": strategies}


@router.post(RoutePaths.STRATEGY)
async def route_create_strategy(body: BodyCreateStrategy):
    with HttpResponseContext():
        id = StrategyQuery.create_entry(body.model_dump())
        return Response(
            content=f"{str(id)}",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )