from fastapi import APIRouter, Depends, Response, status
from context import HttpResponseContext
from api.v1.request_types import BodyCreateTrade, BodyPutTrade
from middleware import api_key_auth
from schema.trade import TradeQuery


router = APIRouter()


class RoutePaths:
    TRADE = "/trade"


@router.get(RoutePaths.TRADE, dependencies=[Depends(api_key_auth)])
async def route_get_trades():
    with HttpResponseContext():
        trades = TradeQuery.get_trades()
        return {"data": trades}


@router.post(RoutePaths.TRADE, dependencies=[Depends(api_key_auth)])
async def route_create_trade(body: BodyCreateTrade):
    with HttpResponseContext():
        id = TradeQuery.create_entry(body.model_dump())
        return Response(
            content=f"{str(id)}",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


@router.put(RoutePaths.TRADE, dependencies=[Depends(api_key_auth)])
async def route_put_trade(body: BodyPutTrade):
    with HttpResponseContext():
        TradeQuery.update_trade(body.id, body.model_dump())
        return Response(
            content="OK",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )
