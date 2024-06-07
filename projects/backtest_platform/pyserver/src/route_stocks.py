import multiprocessing
from fastapi import APIRouter, Response, status

from context import HttpResponseContext
from stock_utils import (
    save_yfinance_historical_klines,
)
from stock_utils import get_nyse_symbols


router = APIRouter()


class RoutePaths:
    FETCH_YFINANCE_DATASET = "/yfinance/dataset/{symbol}"
    GET_SYMBOLS = "/get-symbols"


@router.get(RoutePaths.GET_SYMBOLS)
async def route_fetch_nyse_symbols():
    with HttpResponseContext():
        nyse_symbol_list = get_nyse_symbols()
        return {"data": nyse_symbol_list}


@router.post(RoutePaths.FETCH_YFINANCE_DATASET)
async def route_fetch_yfinance_dataset(symbol: str):
    with HttpResponseContext():
        process = multiprocessing.Process(
            target=save_yfinance_historical_klines, args=(symbol,)
        )
        process.start()
        return Response(
            content="OK", media_type="text/plain", status_code=status.HTTP_200_OK
        )
