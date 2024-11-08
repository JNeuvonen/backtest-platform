from fastapi import APIRouter, Depends, Response, status
from prediction_server.context import HttpResponseContext
from prediction_server.api.v1.request_types import BodyCreateTrade, BodyPutTrade
from prediction_server.middleware import api_key_auth
from common_python.pred_serv_models.cloudlog import slack_log_enter_trade_notif
from common_python.pred_serv_models.trade import TradeQuery
from common_python.pred_serv_models.refetch_strategy_signal import (
    RefetchStrategySignalQuery,
)


router = APIRouter()


class RoutePaths:
    TRADE = "/"


@router.get(RoutePaths.TRADE, dependencies=[Depends(api_key_auth)])
async def route_get_trades():
    with HttpResponseContext():
        trades = TradeQuery.get_trades()
        return {"data": trades}


@router.post(RoutePaths.TRADE, dependencies=[Depends(api_key_auth)])
async def route_create_trade(body: BodyCreateTrade):
    with HttpResponseContext():
        id = TradeQuery.create_entry(body.model_dump())
        if id is not None:
            slack_log_enter_trade_notif(body)
            if body.strategy_id is not None:
                RefetchStrategySignalQuery.create_entry(
                    {"strategy_id": body.strategy_id}
                )
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
