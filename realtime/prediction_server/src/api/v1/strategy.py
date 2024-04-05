from fastapi import APIRouter, Depends, Response, status
from context import HttpResponseContext
from api.v1.request_types import BodyCreateStrategy, BodyPutStrategy
from middleware import api_key_auth
from schema.strategy import StrategyQuery


router = APIRouter()


class RoutePaths:
    STRATEGY = "/"


@router.get(RoutePaths.STRATEGY, dependencies=[Depends(api_key_auth)])
async def route_get_root():
    with HttpResponseContext():
        strategies = StrategyQuery.get_strategies()
        return {"data": strategies}


@router.post(RoutePaths.STRATEGY, dependencies=[Depends(api_key_auth)])
async def route_create_strategy(body: BodyCreateStrategy):
    with HttpResponseContext():
        id = StrategyQuery.create_entry(body.model_dump())
        return Response(
            content=f"{str(id)}",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


@router.put(RoutePaths.STRATEGY, dependencies=[Depends(api_key_auth)])
async def route_put_strategy(body: BodyPutStrategy):
    with HttpResponseContext():
        StrategyQuery.update_strategy(body.id, body.model_dump())
        return Response(
            content="OK",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )
