import asyncio
import pandas as pd
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from backtest_utils import (
    get_backtest_id_to_dataset_name_map,
    get_mass_sim_backtests_equity_curves,
)
from fastapi_utils import convert_to_bool
from query_backtest_history import BacktestHistoryQuery
from constants import BACKTEST_REPORT_HTML_PATH

from context import HttpResponseContext
from manual_backtest import run_manual_backtest, run_rule_based_mass_backtest
from quant_stats_utils import (
    generate_combined_report,
    generate_quant_stats_report_html,
    get_df_returns,
)
from query_backtest import Backtest, BacktestQuery
from query_mass_backtest import MassBacktestQuery
from query_trade import TradeQuery
from request_types import BodyCreateManualBacktest, BodyCreateMassBacktest
from utils import base_model_to_dict, get_periods_per_year


router = APIRouter()


class RoutePaths:
    BACKTEST = "/"
    FETCH_MANY_BACKTESTS = "/fetch-many"
    MASS_BACKTEST = "/mass-backtest"
    MASS_BACKTEST_ID = "/mass-backtest/{mass_backtest_id}"
    DELETE_MANY = "/delete-many"
    BACKTEST_BY_ID = "/{backtest_id}"
    FETCH_BY_DATASET_ID = "/dataset/{dataset_id}"
    DETAILED_SUMMARY = "/{backtest_id}/detailed-summary"
    MASS_BACKTEST_BY_BACKTEST_ID = "/mass-backtest/by-backtest/{backtest_id}"
    COMBINED_STRATEGY_SUMMARY = "/mass-backtest/combined/summary"


@router.get(RoutePaths.BACKTEST_BY_ID)
async def route_get_backtest_by_id(backtest_id):
    with HttpResponseContext():
        backtest = BacktestQuery.fetch_backtest_by_id(backtest_id)

        if backtest is None:
            raise HTTPException(
                detail=f"No backtest found for {backtest_id}", status_code=400
            )

        trades = TradeQuery.fetch_trades_by_backtest_id(backtest.id)
        balance_history = BacktestHistoryQuery.get_entries_by_backtest_id_sorted(
            backtest.id
        )
        return {"data": backtest, "trades": trades, "balance_history": balance_history}


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

        balance_history = BacktestHistoryQuery.get_entries_by_backtest_id_sorted(
            backtest.id
        )
        balance_history = [base_model_to_dict(entry) for entry in balance_history]

        generate_quant_stats_report_html(
            balance_history,
            backtest_info,
            get_periods_per_year(backtest.candle_interval),
        )
        return FileResponse(
            path=BACKTEST_REPORT_HTML_PATH,
            filename="backtest_report_test.html",
            media_type="text/html",
        )


@router.get(RoutePaths.MASS_BACKTEST_BY_BACKTEST_ID)
async def route_mass_backtests_by_backtest_id(backtest_id):
    with HttpResponseContext():
        mass_backtests = MassBacktestQuery.get_mass_backtest_by_original_id(backtest_id)
        return {"data": mass_backtests}


@router.get(RoutePaths.MASS_BACKTEST_ID)
async def route_mass_backtest_by_id(mass_backtest_id):
    with HttpResponseContext():
        mass_backtest = MassBacktestQuery.get_mass_backtest_by_id(mass_backtest_id)

        if mass_backtest is None:
            raise HTTPException(
                detail=f"No mass backtest found for id {mass_backtest_id}",
                status_code=400,
            )

        return {"data": mass_backtest}


@router.get(RoutePaths.BACKTEST)
async def route_fetch_many_backtests(
    list_of_ids: str = Query(...),
    include_equity_curve: Optional[bool] = Query(
        default=None, converter=convert_to_bool
    ),
):
    with HttpResponseContext():
        list_of_ids_arr: List[int] = json.loads(list_of_ids)
        backtests = BacktestQuery.fetch_many_backtests(list_of_ids_arr)
        candle_interval = backtests[0].candle_interval

        equity_curves = []
        datasets_map = {}

        if include_equity_curve is True:
            equity_curves = get_mass_sim_backtests_equity_curves(
                list_of_ids_arr, candle_interval
            )
            datasets_map = get_backtest_id_to_dataset_name_map(list_of_ids_arr)

        return {
            "data": backtests,
            "id_to_dataset_name_map": datasets_map,
            "equity_curves": equity_curves,
        }


@router.get(RoutePaths.COMBINED_STRATEGY_SUMMARY)
async def route_combined_strat_summary(
    list_of_ids: str = Query(...),
):
    with HttpResponseContext():
        list_of_ids_arr: List[int] = json.loads(list_of_ids)
        backtests = BacktestQuery.fetch_many_backtests(list_of_ids_arr)
        candle_interval = backtests[0].candle_interval

        equity_curves = get_mass_sim_backtests_equity_curves(
            list_of_ids_arr, candle_interval
        )

        returns_dict = {}

        datasets_map = get_backtest_id_to_dataset_name_map(list_of_ids_arr)

        for eq_dict in equity_curves:
            for key, value in eq_dict.items():
                df = pd.DataFrame(value)

                df["kline_open_time"] = pd.to_datetime(df["kline_open_time"], unit="ms")
                df.set_index("kline_open_time", inplace=True)
                returns = get_df_returns(df, "portfolio_worth")
                returns += 1
                returns_dict[datasets_map[key]] = returns

        backtest_info = BacktestQuery.fetch_backtest_by_id(list_of_ids_arr[0])

        generate_combined_report(
            returns_dict,
            datasets_map,
            get_periods_per_year(backtest_info.candle_interval),
            backtest_info,
        )

        return FileResponse(
            path=BACKTEST_REPORT_HTML_PATH,
            filename="backtest_report_test.html",
            media_type="text/html",
        )
