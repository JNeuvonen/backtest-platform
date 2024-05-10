import asyncio
from context import HttpResponseContext
from fastapi import APIRouter
from pydantic import BaseModel

from api_binance import (
    get_all_tickers,
    save_historical_klines,
)

router = APIRouter()


class FetchKlinesRequest(BaseModel):
    symbol: str
    interval: str
    use_futures: bool


@router.post("/fetch-klines")
async def get_binance_klines(request: FetchKlinesRequest):
    with HttpResponseContext():
        asyncio.create_task(
            save_historical_klines(
                request.symbol, request.interval, True, request.use_futures
            )
        )
        return {"symbol": request.symbol}


@router.get("/get-all-tickers")
async def get_binance_exchange_info():
    with HttpResponseContext():
        tickers = get_all_tickers()
        return {"pairs": tickers}
