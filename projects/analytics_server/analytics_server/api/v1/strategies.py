from typing import Dict
from common_python.auth.oauth2 import get_user, verify_access_token
from common_python.http_utils import HttpResponse
from common_python.pred_serv_models.strategy import StrategyQuery
from common_python.pred_serv_models.longshortgroup import LongShortGroupQuery
from common_python.pred_serv_models.trade import TradeQuery
from common_python.pred_serv_models.user import User
from common_python.pred_serv_models.longshortticker import LongShortTickerQuery
from common_python.pred_serv_models.strategy_group import StrategyGroupQuery
from common_python.pred_serv_models.longshortpair import LongShortPairQuery
from common_python.pred_serv_models.longshorttrade import LongShortTradeQuery
from common_python.pydantic_models import BodyAnalyticsServiceUpdate
from fastapi import APIRouter, Depends, Response, status


router = APIRouter()


class RoutePaths:
    ROOT = "/"
    STRATEGY_BY_GROUP = "/strategy-group/{group_name}"
    LONGSHORT_BY_GROUP = "/longshort-group/{group_name}"
    DISABLE_LONG_SHORT_STRATEGY = "/longshort-group/{id}/disable"
    DISABLE_STRATEGY = "/strategy-group/{strategy_group_id}/disable"
    ENABLE_STRATEGY = "/strategy-group/{strategy_group_id}/enable"
    UPDATE_MANY = "/update-many"


@router.get(RoutePaths.ROOT)
async def get_strategies(token: str = Depends(verify_access_token)):
    with HttpResponse():
        ls_strategies = LongShortGroupQuery.get_all_strategies()
        directional_strategies = StrategyQuery.get_strategies()
        trades = TradeQuery.get_trades()
        strategy_groups = StrategyGroupQuery.get_all()
        ls_tickers = LongShortTickerQuery.get_all()
        longshort_trades = LongShortTradeQuery.fetch_all()
        pairs = LongShortPairQuery.get_all()

        return {
            "ls_strategies": ls_strategies,
            "ls_tickers": ls_tickers,
            "ls_pairs": pairs,
            "directional_strategies": directional_strategies,
            "trades": trades,
            "strategy_groups": strategy_groups,
            "completed_longshort_trades": longshort_trades,
        }


@router.get(RoutePaths.STRATEGY_BY_GROUP)
async def get_strategies_by_group(
    group_name: str, token: str = Depends(verify_access_token)
):
    with HttpResponse():
        strategy_group = StrategyGroupQuery.get_by_name(group_name)

        if strategy_group is None:
            raise Exception(f"Strategy group was not found for name: {group_name}")

        strategies = StrategyQuery.get_strategies_by_strategy_group_id(
            strategy_group.id
        )
        trades = TradeQuery.fetch_by_strategy_group_id(strategy_group.id)
        return {
            "strategy_group": strategy_group,
            "strategies": strategies,
            "trades": trades,
        }


@router.get(RoutePaths.LONGSHORT_BY_GROUP)
async def get_longshort_strategies_by_group(
    group_name: str, token: str = Depends(verify_access_token)
):
    with HttpResponse():
        longshort_group = LongShortGroupQuery.get_by_name("{SYMBOL}_" + group_name)
        if longshort_group is None:
            raise Exception("No longshort group found for name: " + group_name)

        longshort_tickers = LongShortTickerQuery.get_all_by_group_id(longshort_group.id)
        longshort_pairs = LongShortPairQuery.get_pairs_by_group_id(longshort_group.id)
        longshort_trades = LongShortTradeQuery.fetch_all_by_group_id(longshort_group.id)
        trades = TradeQuery.fetch_by_longshort_group_id(longshort_group.id)

        return {
            "tickers": longshort_tickers,
            "pairs": longshort_pairs,
            "group": longshort_group,
            "completed_trades": longshort_trades,
            "trades": trades,
        }


@router.put(RoutePaths.UPDATE_MANY)
async def put_update_many(
    body: BodyAnalyticsServiceUpdate, user: User = Depends(get_user)
):
    with HttpResponse():
        if user.access_level < 10:
            raise Exception("Authorization failed")

        update_dict: Dict[int, Dict] = {}

        for item in body.strategies:
            update_dict[item.id] = {}
            update_dict[item.id]["allocated_size_perc"] = item.allocation_size_perc
            update_dict[item.id]["should_close_trade"] = bool(item.should_close_trade)
            update_dict[item.id]["is_in_close_only"] = bool(item.is_in_close_only)
            update_dict[item.id]["is_disabled"] = bool(item.is_disabled)
            update_dict[item.id]["stop_processing_new_candles"] = bool(
                item.stop_processing_new_candles
            )
            update_dict[item.id]["id"] = item.id
        StrategyQuery.update_multiple_strategies(update_dict)
        return Response(
            content="OK", media_type="text/plain", status_code=status.HTTP_200_OK
        )


@router.delete(RoutePaths.DISABLE_LONG_SHORT_STRATEGY)
async def longshort_disable_and_close(id: int, user: User = Depends(get_user)):
    with HttpResponse():
        if user.access_level < 10:
            raise Exception("Authorization failed")
        LongShortGroupQuery.update(id, {"is_disabled": True})
        return Response(
            content="OK", media_type="text/plain", status_code=status.HTTP_200_OK
        )


@router.delete(RoutePaths.DISABLE_STRATEGY)
async def route_strategy_disable_and_close(
    strategy_group_id: int, user: User = Depends(get_user)
):
    with HttpResponse():
        if user.access_level < 10:
            raise Exception("Authorization failed")

        strategies = StrategyQuery.get_strategies_by_strategy_group_id(
            strategy_group_id
        )
        update_dict = {}

        for item in strategies:
            update_dict[item.id] = {"is_disabled": True}

        StrategyQuery.update_multiple_strategies(update_dict)
        StrategyGroupQuery.update(strategy_group_id, {"is_disabled": True})
        return Response(
            content="OK", media_type="text/plain", status_code=status.HTTP_200_OK
        )


@router.post(RoutePaths.ENABLE_STRATEGY)
async def route_strategy_enable(strategy_group_id: int, user: User = Depends(get_user)):
    with HttpResponse():
        if user.access_level < 10:
            raise Exception("Authorization failed")

        strategies = StrategyQuery.get_strategies_by_strategy_group_id(
            strategy_group_id
        )
        update_dict = {}

        for item in strategies:
            update_dict[item.id] = {"is_disabled": False}

        StrategyQuery.update_multiple_strategies(update_dict)
        StrategyGroupQuery.update(strategy_group_id, {"is_disabled": False})
        return Response(
            content="OK", media_type="text/plain", status_code=status.HTTP_200_OK
        )
