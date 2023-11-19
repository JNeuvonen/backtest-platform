import threading
import os
from fastapi import APIRouter
from pydantic import BaseModel

from binance_api import save_historical_klines
from log import get_logger

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
