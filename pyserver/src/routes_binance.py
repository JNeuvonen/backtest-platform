import threading
from fastapi import APIRouter
from pydantic import BaseModel

from binance_api import (
    get_all_tickers,
    save_historical_klines,
)

router = APIRouter()


class FetchKlinesRequest(BaseModel):
    symbol: str
    interval: str


@router.post("/fetch-klines")
async def get_binance_klines(request: FetchKlinesRequest):
    worker_thread = threading.Thread(
        target=save_historical_klines, args=(request.symbol, request.interval)
    )
    worker_thread.start()
    return {"symbol": request.symbol}


@router.get("/get-all-tickers")
async def get_binance_exchange_info():
    tickers = get_all_tickers()
    return {"pairs": tickers}
