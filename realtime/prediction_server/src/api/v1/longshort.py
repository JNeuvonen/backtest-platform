from fastapi import APIRouter, Depends, Response, status

from middleware import api_key_auth
from api.v1.request_types import (
    BodyCreateLongShortStrategy,
    EnterLongShortPairBody,
    ExitLongShortPairBody,
    UpdateLongShortPairBody,
)
from context import HttpResponseContext
from trade_utils import enter_longshort_trade, exit_longshort_trade
from schema.longshortpair import LongShortPairQuery
from schema.data_transformation import DataTransformationQuery
from schema.longshortgroup import LongShortGroupQuery
from schema.longshortticker import LongShortTickerQuery
from binance_utils import infer_assets


router = APIRouter()


class RoutePaths:
    LONG_SHORT = "/"
    LONG_SHORT_TICKERS = "/tickers/{longshort_group_id}"
    LONG_SHORT_PAIRS = "/pairs/{longshort_group_id}"
    LONG_SHORT_PAIR = "/pair/{pair_id}"
    LONG_SHORT_PAIR_ENTER = "/pair/{pair_id}/enter"
    LONG_SHORT_PAIR_EXIT = "/pair/{pair_id}/exit"


@router.post(RoutePaths.LONG_SHORT, dependencies=[Depends(api_key_auth)])
async def route_create_long_short_strategy(body: BodyCreateLongShortStrategy):
    with HttpResponseContext():
        if "{SYMBOL}" not in body.name:
            raise Exception("Strategy must contain string {SYMBOL}")

        long_short_group_id = LongShortGroupQuery.create_entry(
            {
                "name": body.name,
                "candle_interval": body.candle_interval,
                "buy_cond": body.buy_cond,
                "sell_cond": body.sell_cond,
                "exit_cond": body.exit_cond,
                "num_req_klines": body.num_req_klines,
                "max_simultaneous_positions": body.max_simultaneous_positions,
                "klines_until_close": body.klines_until_close,
                "kline_size_ms": body.kline_size_ms,
                "max_leverage_ratio": body.max_leverage_ratio,
                "take_profit_threshold_perc": body.take_profit_threshold_perc,
                "stop_loss_threshold_perc": body.stop_loss_threshold_perc,
                "use_time_based_close": body.use_time_based_close,
                "use_profit_based_close": body.use_profit_based_close,
                "use_stop_loss_based_close": body.use_stop_loss_based_close,
                "use_taker_order": body.use_taker_order,
            }
        )

        for item in body.data_transformations:
            DataTransformationQuery.create_transformation(
                {
                    "transformation_code": item.transformation_code,
                    "long_short_group_id": long_short_group_id,
                }
            )

        strategy_name = body.name

        for item in body.asset_universe:
            quantity_precision = item["tradeQuantityPrecision"]
            symbol = item["symbol"]
            dataset_name = strategy_name.format(SYMBOL=symbol)
            assets = infer_assets(symbol)
            quote_asset = assets["quoteAsset"]
            base_asset = assets["baseAsset"]
            LongShortTickerQuery.create_entry(
                {
                    "dataset_name": dataset_name,
                    "symbol": symbol,
                    "quote_asset": quote_asset,
                    "base_asset": base_asset,
                    "trade_quantity_precision": quantity_precision,
                    "long_short_group_id": long_short_group_id,
                }
            )
        LongShortGroupQuery.update(long_short_group_id, {"is_disabled": False})

        return Response(
            content=f"{str(long_short_group_id)}",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


@router.get(RoutePaths.LONG_SHORT, dependencies=[Depends(api_key_auth)])
async def route_get_longshort_strategies():
    with HttpResponseContext():
        longshort_strategies = LongShortGroupQuery.get_strategies()
        return {"data": longshort_strategies}


@router.get(RoutePaths.LONG_SHORT_TICKERS, dependencies=[Depends(api_key_auth)])
async def route_get_longshort_tickers_by_group(longshort_group_id):
    with HttpResponseContext():
        tickers = LongShortTickerQuery.get_all_by_group_id(longshort_group_id)
        return {"data": tickers}


@router.get(RoutePaths.LONG_SHORT_PAIRS, dependencies=[Depends(api_key_auth)])
async def route_get_longshort_pairs_by_group(longshort_group_id):
    with HttpResponseContext():
        tickers = LongShortPairQuery.get_pairs_by_group_id(longshort_group_id)
        return {"data": tickers}


@router.put(RoutePaths.LONG_SHORT_PAIR, dependencies=[Depends(api_key_auth)])
async def route_put_longshort_pair(pair_id: int, body: UpdateLongShortPairBody):
    with HttpResponseContext():
        LongShortPairQuery.update_entry(pair_id, body.model_dump())
        return Response(
            content="OK",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


@router.post(RoutePaths.LONG_SHORT_PAIR_ENTER, dependencies=[Depends(api_key_auth)])
async def route_post_longshort_enter(pair_id: int, body: EnterLongShortPairBody):
    with HttpResponseContext():
        longshort_pair = LongShortPairQuery.get_pair_by_id(pair_id)
        enter_longshort_trade(longshort_pair, body)
        return Response(
            content="OK",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


@router.post(RoutePaths.LONG_SHORT_PAIR_EXIT, dependencies=[Depends(api_key_auth)])
async def route_post_longshort_exit(pair_id: int, body: ExitLongShortPairBody):
    with HttpResponseContext():
        longshort_pair = LongShortPairQuery.get_pair_by_id(pair_id)

        if longshort_pair is None:
            raise Exception(f"Longshort pair was not found for id {pair_id}")

        exit_longshort_trade(longshort_pair, body)
        return Response(
            content="OK",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )
