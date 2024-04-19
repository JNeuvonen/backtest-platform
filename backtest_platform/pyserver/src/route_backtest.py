import asyncio
import json
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from constants import BACKTEST_REPORT_HTML_PATH

from context import HttpResponseContext
from db import timedelta_to_candlesize
from manual_backtest import run_manual_backtest, run_rule_based_mass_backtest
from quant_stats_utils import (
    generate_quant_stats_report_html,
    update_backtest_report_html,
)
from query_backtest import BacktestQuery
from query_mass_backtest import MassBacktestQuery
from query_trade import TradeQuery
from request_types import BodyCreateManualBacktest, BodyCreateMassBacktest
from utils import get_periods_per_year


router = APIRouter()


class RoutePaths:
    BACKTEST = "/"
    MASS_BACKTEST = "/mass-backtest"
    DELETE_MANY = "/delete-many"
    BACKTEST_BY_ID = "/{backtest_id}"
    FETCH_BY_DATASET_ID = "/dataset/{dataset_id}"
    DETAILED_SUMMARY = "/{backtest_id}/detailed-summary"


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

        candle_size = original_backtest.candle_interval

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


@router.get(RoutePaths.DETAILED_SUMMARY)
async def route_detailed_summary(backtest_id):
    with HttpResponseContext():
        backtest = BacktestQuery.fetch_backtest_by_id(backtest_id)
        backtest_info = BodyCreateManualBacktest(
            backtest_data_range=[
                backtest.backtest_range_start,
                backtest.backtest_range_end,
            ],
            open_trade_cond=backtest.open_trade_cond,
            close_trade_cond=backtest.close_trade_cond,
            is_short_selling_strategy=backtest.is_short_selling_strategy,
            use_stop_loss_based_close=backtest.use_stop_loss_based_close,
            use_time_based_close=backtest.use_time_based_close,
            use_profit_based_close=backtest.use_profit_based_close,
            dataset_id=backtest.dataset_id,
            trading_fees_perc=backtest.trading_fees_perc,
            slippage_perc=backtest.slippage_perc,
            short_fee_hourly=backtest.short_fee_hourly,
            take_profit_threshold_perc=backtest.take_profit_threshold_perc,
            stop_loss_threshold_perc=backtest.stop_loss_threshold_perc,
            name=backtest.name,
            klines_until_close=backtest.klines_until_close,
        )

        generate_quant_stats_report_html(
            backtest.data, backtest_info, get_periods_per_year(backtest.candle_interval)
        )
        return FileResponse(
            path=BACKTEST_REPORT_HTML_PATH,
            filename="backtest_report_test.html",
            media_type="text/html",
        )
