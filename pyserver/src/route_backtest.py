import json
from typing import List
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from context import HttpResponseContext
from manual_backtest import run_manual_backtest
from query_backtest import BacktestQuery
from query_trade import TradeQuery
from request_types import BodyCreateManualBacktest, BodyDeleteManyBacktestsById


router = APIRouter()


class RoutePaths:
    BACKTEST = "/"
    DELETE_MANY = "/delete-many"
    BACKTEST_BY_ID = "/{backtest_id}"
    FETCH_BY_DATASET_ID = "/dataset/{dataset_id}"


@router.get(RoutePaths.BACKTEST_BY_ID)
async def route_get_backtest_by_id(backtest_id):
    with HttpResponseContext():
        backtest = BacktestQuery.fetch_backtest_by_id(backtest_id)
        if backtest is None:
            raise HTTPException(
                detail=f"No backtest found for {backtest_id}", status_code=400
            )

        trades = TradeQuery.fetch_trades_by_backtest_id(backtest.id)
        return {"data": backtest, "trades": trades}


@router.post(RoutePaths.BACKTEST)
async def route_create_manual_backtest(body: BodyCreateManualBacktest):
    with HttpResponseContext():
        backtest = run_manual_backtest(body)
        return {"data": backtest}


@router.get(RoutePaths.FETCH_BY_DATASET_ID)
async def route_fetch_by_dataset_id(dataset_id):
    with HttpResponseContext():
        backtests = BacktestQuery.fetch_backtests_by_dataset_id(dataset_id)
        return {"data": backtests}


@router.delete(RoutePaths.DELETE_MANY)
async def route_delete_many(list_of_ids: str = Query(...)):
    with HttpResponseContext():
        BacktestQuery.delete_backtests_by_ids(json.loads(list_of_ids))
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )
