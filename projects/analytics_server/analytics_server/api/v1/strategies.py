from common_python.auth.oauth2 import verify_access_token
from common_python.http_utils import HttpResponse
from common_python.pred_serv_models.strategy import StrategyQuery
from common_python.pred_serv_models.longshortgroup import LongShortGroupQuery
from common_python.pred_serv_models.trade import TradeQuery
from common_python.pred_serv_orm import StrategyGroup
from common_python.pred_serv_models.longshortticker import LongShortTickerQuery
from common_python.pred_serv_models.strategy_group import StrategyGroupQuery
from common_python.pred_serv_models.longshortpair import LongShortPairQuery
from fastapi import APIRouter, Depends


router = APIRouter()


class RoutePaths:
    ROOT = "/"


@router.get(RoutePaths.ROOT)
async def get_strategies(token: str = Depends(verify_access_token)):
    with HttpResponse():
        ls_strategies = LongShortGroupQuery.get_all_strategies()
        directional_strategies = StrategyQuery.get_strategies()
        trades = TradeQuery.get_trades()
        strategy_groups = StrategyGroupQuery.get_all()
        ls_tickers = LongShortTickerQuery.get_all()
        pairs = LongShortPairQuery.get_all()

        return {
            "ls_strategies": ls_strategies,
            "ls_tickers": ls_tickers,
            "ls_pairs": pairs,
            "directional_strategies": directional_strategies,
            "trades": trades,
            "strategy_groups": strategy_groups,
        }
