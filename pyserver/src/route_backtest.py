from fastapi import APIRouter

from context import HttpResponseContext
from manual_backtest import run_manual_backtest
from query_backtest import BacktestQuery
from request_types import BodyCreateManualBacktest


router = APIRouter()


class RoutePaths:
    BACKTEST = "/"
    FETCH_BY_DATASET_ID = "/dataset/{dataset_id}"


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
