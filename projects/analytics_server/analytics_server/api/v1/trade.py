from common_python.auth.oauth2 import verify_access_token
from common_python.http_utils import HttpResponse
from common_python.pred_serv_models.trade import TradeQuery
from fastapi import APIRouter, Depends, Response, status


router = APIRouter()


class RoutePaths:
    ROOT = "/"
    GET_COMPLETED_TRADES = "/completed"
    GET_UNCOMPLETED_TRADES = "/uncompleted"


@router.get(RoutePaths.GET_COMPLETED_TRADES)
async def get_completed_trades(token: str = Depends(verify_access_token)):
    with HttpResponse():
        trades = TradeQuery.get_completed_trades()
        return {"data": trades}


@router.get(RoutePaths.GET_UNCOMPLETED_TRADES)
async def get_uncompleted_trades(token: str = Depends(verify_access_token)):
    with HttpResponse():
        trades = TradeQuery.get_uncompleted_trades()
        return {"data": trades}
