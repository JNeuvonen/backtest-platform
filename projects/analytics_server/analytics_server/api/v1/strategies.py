from common_python.auth.oauth2 import verify_access_token
from common_python.http_utils import HttpResponse
from common_python.pred_serv_models.strategy import StrategyQuery
from common_python.pred_serv_models.longshortgroup import LongShortGroupQuery
from fastapi import APIRouter, Depends


router = APIRouter()


class RoutePaths:
    ROOT = "/"


@router.get(RoutePaths.ROOT)
async def get_strategies(token: str = Depends(verify_access_token)):
    with HttpResponse():
        ls_strategies = LongShortGroupQuery.get_all_strategies()
        directional_strategies = StrategyQuery.get_strategies()
        return {
            "ls_strategies": ls_strategies,
            "directional_strategies": directional_strategies,
        }
