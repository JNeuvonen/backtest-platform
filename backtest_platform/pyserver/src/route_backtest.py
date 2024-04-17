import asyncio
import json
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from context import HttpResponseContext
from db import timedelta_to_candlesize
from manual_backtest import run_manual_backtest, run_rule_based_mass_backtest
from query_backtest import BacktestQuery
from query_mass_backtest import MassBacktestQuery
from query_trade import TradeQuery
from request_types import BodyCreateManualBacktest, BodyCreateMassBacktest


router = APIRouter()


class RoutePaths:
    BACKTEST = "/"
    MASS_BACKTEST = "/mass-backtest"
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


@router.post(RoutePaths.MASS_BACKTEST)
async def route_mass_backtest(body: BodyCreateMassBacktest):
    with HttpResponseContext():
        original_backtest = BacktestQuery.fetch_backtest_by_id(
            body.original_backtest_id
        )

        if original_backtest is None:
            raise HTTPException(
                detail=f"No backtest found for {body.original_backtest_id}",
                status_code=400,
            )

        candle_size = timedelta_to_candlesize(
            original_backtest.kline_time_delta_ms / 1000
        )

        mass_backtest_id = MassBacktestQuery.create_entry(
            {"original_backtest_id": body.original_backtest_id}
        )

        asyncio.create_task(
            run_rule_based_mass_backtest(
                mass_backtest_id, body, candle_size, original_backtest
            )
        )

        return Response(
            content=str(mass_backtest_id),
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


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
