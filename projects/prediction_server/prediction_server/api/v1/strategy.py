import copy
from common_python.pred_serv_models.refetch_strategy_signal import (
    RefetchStrategySignalQuery,
)
from fastapi import APIRouter, Depends, HTTPException, Response, status
from context import HttpResponseContext
from api.v1.request_types import (
    BodyCreateStrategy,
    BodyCreateStrategyGroup,
    BodyPutStrategy,
    BodyUpdateStrategy,
    BodyUpdateTradeClose,
)
from middleware import api_key_auth
from common_python.pred_serv_models.strategy import StrategyQuery
from common_python.pred_serv_models.trade import TradeQuery
from common_python.pred_serv_models.strategy_group import StrategyGroupQuery
from log import get_logger
from common_python.pred_serv_models.data_transformation import DataTransformationQuery
from prediction_server.strategy import get_strategy_name
from trade_utils import close_long_trade, close_short_trade, update_strategy_state


router = APIRouter()


class RoutePaths:
    STRATEGY = "/"
    UPDATE_TRADE_CLOSE = "/{id}/close-trade"
    UPDATE_STRATEGY = "/{id}"
    CREATE_STRATEGY_GROUP = "/strategy-group"


@router.get(RoutePaths.STRATEGY, dependencies=[Depends(api_key_auth)])
async def route_get_root():
    with HttpResponseContext():
        strategies = StrategyQuery.get_strategies()
        return {"data": strategies}


@router.post(RoutePaths.STRATEGY, dependencies=[Depends(api_key_auth)])
async def route_create_strategy(body: BodyCreateStrategy):
    with HttpResponseContext():
        data_transformations = body.data_transformations
        body_copy = body.model_dump(exclude={"data_transformations"})

        id = StrategyQuery.create_entry(body_copy)

        for item in data_transformations:
            DataTransformationQuery.create_transformation(
                {"transformation_code": item.transformation_code, "strategy_id": id}
            )
        return Response(
            content=f"{str(id)}",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


@router.post(RoutePaths.CREATE_STRATEGY_GROUP, dependencies=[Depends(api_key_auth)])
async def route_create_strategy_group(body: BodyCreateStrategyGroup):
    with HttpResponseContext():
        data_transformations = body.data_transformations

        transformation_ids = []

        strategy_group_id = StrategyGroupQuery.create_entry(
            {
                "name": body.strategy_group,
                "is_auto_adaptive_group": body.is_auto_adaptive_group,
                "num_symbols_for_auto_adaptive": body.num_symbols_for_auto_adaptive,
                "enter_trade_code": body.enter_trade_code,
                "exit_trade_code": body.exit_trade_code,
                "fetch_datasources_code": body.fetch_datasources_code,
                "candle_interval": body.candle_interval,
                "priority": body.priority,
                "num_req_klines": body.num_req_klines,
                "kline_size_ms": body.kline_size_ms,
                "minimum_time_between_trades_ms": body.minimum_time_between_trades_ms,
                "maximum_klines_hold_time": body.maximum_klines_hold_time,
                "allocated_size_perc": body.allocated_size_perc,
                "take_profit_threshold_perc": body.take_profit_threshold_perc,
                "stop_loss_threshold_perc": body.stop_loss_threshold_perc,
                "use_time_based_close": body.use_time_based_close,
                "use_profit_based_close": body.use_profit_based_close,
                "use_stop_loss_based_close": body.use_stop_loss_based_close,
                "use_taker_order": body.use_taker_order,
                "should_calc_stops_on_pred_serv": body.should_calc_stops_on_pred_serv,
                "is_leverage_allowed": body.is_leverage_allowed,
                "is_short_selling_strategy": body.is_short_selling_strategy,
            }
        )
        for item in data_transformations:
            id = DataTransformationQuery.create_transformation(
                {
                    "transformation_code": item.transformation_code,
                    "strategy_group_id": strategy_group_id,
                }
            )
            transformation_ids.append(id)

        symbols = body.symbols

        body_copy = body.model_dump(
            exclude={
                "data_transformations",
                "symbols",
                "num_symbols_for_auto_adaptive",
                "is_auto_adaptive_group",
            }
        )

        for item in symbols:
            try:
                body_copy_clone = copy.deepcopy(body_copy)
                body_copy_clone["name"] = get_strategy_name(
                    body.strategy_group, item.symbol
                )
                body_copy_clone["symbol"] = item.symbol
                body_copy_clone["base_asset"] = item.base_asset
                body_copy_clone["quote_asset"] = item.quote_asset
                body_copy_clone[
                    "trade_quantity_precision"
                ] = item.trade_quantity_precision
                body_copy_clone["strategy_group_id"] = strategy_group_id
                StrategyQuery.create_entry(body_copy_clone)
            except Exception as err:
                print(err)
                pass

        StrategyGroupQuery.update(
            strategy_group_id,
            {"transformation_ids": transformation_ids, "is_disabled": False},
        )

        return Response(
            content=f"{strategy_group_id}",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


@router.put(RoutePaths.STRATEGY, dependencies=[Depends(api_key_auth)])
async def route_put_strategy(body: BodyPutStrategy):
    with HttpResponseContext():
        body_json = body.model_dump()

        if body.remaining_position_on_trade is not None:
            logger = get_logger()
            logger.info(f"Entering trade with payload: {body_json}")

        StrategyQuery.update_strategy(body.id, body_json)
        return Response(
            content="OK",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


@router.put(RoutePaths.UPDATE_TRADE_CLOSE, dependencies=[Depends(api_key_auth)])
async def route_update_trade_close(id: int, body: BodyUpdateTradeClose):
    with HttpResponseContext():
        strategy = StrategyQuery.get_strategy_by_id(id)

        if strategy is None:
            raise HTTPException(
                status_code=400, detail=f"Strategy was not found with id {id}"
            )

        trade = TradeQuery.get_trade_by_id(strategy.active_trade_id)

        if trade is None:
            raise HTTPException(
                status_code=400,
                detail=f"Trade was not found with id {strategy.active_trade_id}",
            )

        update_strategy_state(strategy, body)

        if strategy.is_short_selling_strategy is True:
            close_short_trade(strategy, trade, body)

        else:
            close_long_trade(strategy, trade, body)
