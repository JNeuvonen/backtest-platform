from fastapi import APIRouter, Depends, HTTPException, Response, status
from context import HttpResponseContext
from api.v1.request_types import (
    BodyCreateStrategy,
    BodyPutStrategy,
    BodyUpdateTradeClose,
)
from middleware import api_key_auth
from schema.strategy import StrategyQuery
from schema.trade import TradeQuery
from trade_utils import close_long_trade, close_short_trade, update_strategy_state


router = APIRouter()


class RoutePaths:
    STRATEGY = "/"
    UPDATE_TRADE_CLOSE = "/{id}/close-trade"


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


@router.put(RoutePaths.UPDATE_TRADE_CLOSE, dependencies=[Depends(api_key_auth)])
async def route_update_trade_close(id: int, body: BodyUpdateTradeClose):
    with HttpResponseContext():
        strategy = StrategyQuery.get_strategy_by_id(id)

        if strategy is None:
            raise HTTPException(
                status_code=400, detail=f"Strategy was not found with id {id}"
            )

        trade = TradeQuery.get_trade_by_id(strategy.active_trade_id)

        if trade is None:
            raise HTTPException(
                status_code=400,
                detail=f"Trade was not found with id {strategy.active_trade_id}",
            )

        update_strategy_state(strategy, body)

        if strategy.is_short_selling_strategy is True:
            close_short_trade(strategy, trade, body)

        else:
            close_long_trade(strategy, trade, body)
