import asyncio
import pandas as pd
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from mass_rule_based_backtest import run_rule_based_backtest_on_universe
from query_data_transformation import DataTransformationQuery
from query_symbol import SymbolQuery
from backtest_utils import (
    get_backtest_id_to_dataset_name_map,
    get_mass_sim_backtests_equity_curves,
)
from config import is_testing
from fastapi_utils import convert_to_bool
from long_short_backtest import run_long_short_backtest
from ml_based_backtest import (
    run_ml_based_backtest,
)
from query_backtest_history import BacktestHistoryQuery
from constants import BACKTEST_REPORT_HTML_PATH

from context import HttpResponseContext
from manual_backtest import run_manual_backtest, run_rule_based_mass_backtest
from quant_stats_utils import (
    generate_combined_report,
    generate_quant_stats_report_html,
    get_df_returns,
)
from query_backtest import BacktestQuery
from query_mass_backtest import MassBacktestQuery
from query_pair_trade import PairTradeQuery
from query_trade import TradeQuery
from multiprocessing import Process
from multiprocess import log_event_queue
from request_types import (
    BodyCreateLongShortBacktest,
    BodyCreateManualBacktest,
    BodyCreateMassBacktest,
    BodyMLBasedBacktest,
    BodyRuleBasedMultiStrategy,
    BodyRuleBasedOnUniverse,
)
from rule_based_multistrat import run_rule_based_multistrat_backtest
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
    MASS_SIM_DETAILED_SUMMARY = "/multistrategy/detailed-summary/{backtest_id}"
    MASS_BACKTEST_BY_BACKTEST_ID = "/mass-backtest/by-backtest/{backtest_id}"
    COMBINED_STRATEGY_SUMMARY = "/mass-backtest/combined/summary"
    LONG_SHORT_BACKTEST = "/long-short-backtest"
    FETCH_LONG_SHORT_BACKTEST = "/mass-backtest/long-short/fetch"
    ML_BASED_BACKTEST = "/ml-based"
    FETCH_MASS_BACKTEST_SYMBOLS = "/mass-backtest/long-short/symbols/{backtest_id}"
    FETCH_MASS_BACKTEST_TRANSFORMATIONS = (
        "/mass-backtest/long-short/transformations/{backtest_id}"
    )
    RULE_BASED_BACKTEST_ON_UNIVERSE = "/rule-based-on-universe"
    RULE_BASED_MULTISTRAT = "/rule-based-multi-strat"
    FETCH_RULE_BASED_MASSBACKTESTS = "/mass-backtest/rule-based/on-asset-universe"
    FETCH_MULTI_STRAT_BACKTESTS = "/mass-backtest/rule-based/multi-strat"


@router.get(RoutePaths.BACKTEST_BY_ID)
async def route_get_backtest_by_id(backtest_id):
    with HttpResponseContext():
        backtest = BacktestQuery.fetch_backtest_by_id(backtest_id)

        if backtest is None:
            raise HTTPException(
                detail=f"No backtest found for {backtest_id}", status_code=400
            )

        trades = TradeQuery.fetch_trades_by_backtest_id(backtest["id"])
        balance_history = BacktestHistoryQuery.get_entries_by_backtest_id_sorted(
            backtest["id"]
        )
        pair_trades = []

        if backtest["is_long_short_strategy"] is True:
            pair_trades = PairTradeQuery.fetch_all_by_backtest_id(backtest_id)

        return {
            "data": backtest,
            "trades": trades,
            "balance_history": balance_history,
            "pair_trades": pair_trades,
        }


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

        candle_size = original_backtest["candle_interval"]

        mass_backtest_id = MassBacktestQuery.create_entry(
            {"original_backtest_id": body.original_backtest_id}
        )

        process = Process(
            target=run_rule_based_mass_backtest,
            args=(
                log_event_queue,
                mass_backtest_id,
                body,
                candle_size,
                original_backtest,
            ),
        )
        process.start()

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
                backtest["backtest_range_start"],
                backtest["backtest_range_end"],
            ],
            open_trade_cond=backtest["open_trade_cond"],
            close_trade_cond=backtest["close_trade_cond"],
            is_short_selling_strategy=backtest["is_short_selling_strategy"],
            use_stop_loss_based_close=backtest["use_stop_loss_based_close"],
            use_time_based_close=backtest["use_time_based_close"],
            use_profit_based_close=backtest["use_profit_based_close"],
            dataset_id=backtest["dataset_id"],
            trading_fees_perc=backtest["trading_fees_perc"],
            slippage_perc=backtest["slippage_perc"],
            short_fee_hourly=backtest["short_fee_hourly"],
            take_profit_threshold_perc=backtest["take_profit_threshold_perc"],
            stop_loss_threshold_perc=backtest["stop_loss_threshold_perc"],
            name=backtest["name"],
            klines_until_close=backtest["klines_until_close"],
        )

        balance_history = BacktestHistoryQuery.get_entries_by_backtest_id_sorted(
            backtest["id"]
        )
        balance_history = [base_model_to_dict(entry) for entry in balance_history]

        generate_quant_stats_report_html(
            balance_history,
            backtest_info,
            get_periods_per_year(backtest["candle_interval"]),
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
        candle_interval = backtests[0]["candle_interval"]

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
        candle_interval = backtests[0]["candle_interval"]

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
            365,
            backtest_info,
        )

        return FileResponse(
            path=BACKTEST_REPORT_HTML_PATH,
            filename="backtest_report_test.html",
            media_type="text/html",
        )


@router.post(RoutePaths.LONG_SHORT_BACKTEST)
async def route_long_short_backtest(body: BodyCreateLongShortBacktest):
    with HttpResponseContext():
        process = Process(target=run_long_short_backtest, args=(body,))
        process.start()
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.get(RoutePaths.FETCH_LONG_SHORT_BACKTEST)
async def route_get_long_short_backtests():
    with HttpResponseContext():
        backtests = BacktestQuery.fetch_all_long_short_backtests()
        return {"data": backtests}


@router.post(RoutePaths.ML_BASED_BACKTEST)
async def route_create_ml_based_backtest(body: BodyMLBasedBacktest):
    with HttpResponseContext():
        if is_testing():
            await run_ml_based_backtest((body))
        else:
            asyncio.create_task(run_ml_based_backtest(body))
        pass


@router.get(RoutePaths.FETCH_MASS_BACKTEST_SYMBOLS)
async def route_fetch_mass_backtest_symbols(backtest_id: int):
    with HttpResponseContext():
        symbols = SymbolQuery.get_symbols_by_backtest(backtest_id)
        return {"data": symbols}


@router.get(RoutePaths.FETCH_MASS_BACKTEST_TRANSFORMATIONS)
async def route_fetch_mass_backtest_transformations(backtest_id: int):
    with HttpResponseContext():
        transformations = DataTransformationQuery.get_transformations_by_backtest(
            backtest_id
        )
        return {"data": transformations}


@router.post(RoutePaths.RULE_BASED_BACKTEST_ON_UNIVERSE)
async def route_rule_based_on_universe(body: BodyRuleBasedOnUniverse):
    with HttpResponseContext():
        if is_testing():
            run_rule_based_backtest_on_universe(log_event_queue, body)
        else:
            process = Process(
                target=run_rule_based_backtest_on_universe,
                args=(
                    log_event_queue,
                    body,
                ),
            )
            process.start()
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.get(RoutePaths.FETCH_RULE_BASED_MASSBACKTESTS)
async def route_fetch_rule_based_massbacktests():
    with HttpResponseContext():
        backtests = BacktestQuery.fetch_all_rule_based_mass_backtests()
        return {"data": backtests}


@router.post(RoutePaths.RULE_BASED_MULTISTRAT)
async def route_rule_based_multistrat(body: BodyRuleBasedMultiStrategy):
    with HttpResponseContext():
        if is_testing():
            run_rule_based_multistrat_backtest(log_event_queue, body)
        else:
            process = Process(
                target=run_rule_based_multistrat_backtest,
                args=(
                    log_event_queue,
                    body,
                ),
            )
            process.start()

        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.get(RoutePaths.FETCH_MULTI_STRAT_BACKTESTS)
async def route_fetch_multistrat_backtests():
    with HttpResponseContext():
        backtests = BacktestQuery.fetch_all_multistrat_backtests()
        return {"data": backtests}


@router.get(RoutePaths.MASS_SIM_DETAILED_SUMMARY)
async def route_mass_sim_detailed_summary(backtest_id: int):
    with HttpResponseContext():
        balance_history = BacktestHistoryQuery.get_entries_by_backtest_id_sorted(
            backtest_id
        )
        balance_history = [base_model_to_dict(entry) for entry in balance_history]
        for item in balance_history:
            item["kline_open_time"] = item["kline_open_time"] * 1000
        generate_quant_stats_report_html(balance_history, None, 365)

        return FileResponse(
            path=BACKTEST_REPORT_HTML_PATH,
            filename="backtest_report_test.html",
            media_type="text/html",
        )
