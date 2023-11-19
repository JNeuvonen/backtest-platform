from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class FetchKlinesRequest(BaseModel):
    symbol: str
    interval: str


@router.post("/fetch-klines")
async def get_binance_klines(request: FetchKlinesRequest):
    return {"symbol": request.symbol}
