from common_python.pred_serv_models.data_transformation import DataTransformationQuery
from common_python.pred_serv_models.strategy_group import (
    StrategyGroup,
    StrategyGroupQuery,
)
import uvicorn
import os
from multiprocessing import Process, Event
from contextlib import asynccontextmanager

from common_python.pred_serv_orm import create_tables, test_db_conn
from common_python.pred_serv_models.strategy import StrategyQuery
from common_python.pred_serv_models.trade_info_tick import TradeInfoTickQuery
from common_python.pred_serv_models.refetch_strategy_signal import (
    RefetchStrategySignal,
)
from common_python.pred_serv_models.longshortgroup import LongShortGroupQuery
from common_python.server_config import get_service_port

from prediction_server.config import get_auto_whitelisted_ip, run_background_processes
from fastapi import FastAPI, Response, status
from prediction_server.log import LogExceptionContext, get_logger
from prediction_server.api.v1.strategy import router as v1_strategy_router
from prediction_server.api.v1.log import router as v1_cloudlogs_router
from prediction_server.api.v1.account import router as v1_account_router
from prediction_server.api.v1.trade import router as v1_trade_router
from prediction_server.api.v1.api_key import router as v1_api_key_router
from prediction_server.api.v1.longshort import router as v1_longshort
from prediction_server.middleware import ValidateIPMiddleware
from prediction_server.constants import LogLevel
from prediction_server.binance_utils import (
    gen_trading_decisions,
    update_long_short_enters,
    update_long_short_exits,
)
from prediction_server.rule_based_loop_utils import RuleBasedLoopManager
from prediction_server.utils import get_current_timestamp_ms
from prediction_server.strategy import (
    format_pair_trade_loop_msg,
    format_pred_loop_log_msg,
    update_strategies_state_dict,
)
from common_python.pred_serv_models.whitelisted_ip import WhiteListedIPQuery
from common_python.pred_serv_models.cloudlog import CloudLogQuery, create_log
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    WhiteListedIPQuery.create_entry({"ip": get_auto_whitelisted_ip()})
    yield


class Routers:
    V1_STRATEGY = "/v1/strategy"
    V1_LOGS = "/v1/log"
    V1_ACC = "/v1/acc"
    V1_TRADE = "/v1/trade"
    V1_API_KEY = "/v1/api-key"
    V1_LONG_SHORT = "/v1/longshort"


app = FastAPI(lifespan=lifespan)

app.include_router(v1_cloudlogs_router, prefix=Routers.V1_LOGS)
app.include_router(v1_strategy_router, prefix=Routers.V1_STRATEGY)
app.include_router(v1_account_router, prefix=Routers.V1_ACC)
app.include_router(v1_trade_router, prefix=Routers.V1_TRADE)
app.include_router(v1_api_key_router, prefix=Routers.V1_API_KEY)
app.include_router(v1_longshort, prefix=Routers.V1_LONG_SHORT)

app.add_middleware(ValidateIPMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def rule_based_loop(stop_event):
    logger = get_logger()
    logger.info("Starting rule based loop")

    last_loop_complete_timestamp = get_current_timestamp_ms()
    last_slack_message_timestamp = get_current_timestamp_ms()
    last_logs_cleared_timestamp = get_current_timestamp_ms()
    last_longest_loop_completion_time_reset = get_current_timestamp_ms()

    longest_loop_completion_time = 0

    state_manager = RuleBasedLoopManager()

    while not stop_event.is_set():
        with LogExceptionContext(re_raise=False):
            current_state_dict = {
                "active_strategies": 0,
                "strats_on_error": 0,
                "strats_in_pos": 0,
                "strats_wanting_to_close": 0,
                "strats_wanting_to_enter": 0,
                "strats_still": 0,
            }
            strategies_info = []

            for strategy in state_manager.strategies:
                if strategy.stop_processing_new_candles is True:
                    continue

                trading_decisions = gen_trading_decisions(strategy, state_manager)
                update_strategies_state_dict(
                    strategy, trading_decisions, current_state_dict, strategies_info
                )

            if get_current_timestamp_ms() >= last_slack_message_timestamp + 60000:
                create_log(
                    msg=format_pred_loop_log_msg(
                        current_state_dict,
                        strategies_info,
                        last_loop_complete_timestamp,
                        longest_loop_completion_time,
                    ),
                    level=LogLevel.INFO,
                )
                last_slack_message_timestamp = get_current_timestamp_ms()

            if (
                get_current_timestamp_ms()
                >= last_longest_loop_completion_time_reset + 60000 * 60 * 64
            ):
                longest_loop_completion_time = 0

            if get_current_timestamp_ms() >= last_logs_cleared_timestamp + 60000 * 60:
                CloudLogQuery.clear_outdated_logs()
                last_logs_cleared_timestamp = get_current_timestamp_ms()

            state_manager.update_changed_strategies()
            state_manager.update_local_state()
            state_manager.check_refetch_signal()
            state_manager.create_trade_info_ticks()
            state_manager.update_strategy_groups()

            loop_complete_time_ms = (
                get_current_timestamp_ms() - last_loop_complete_timestamp
            )

            if loop_complete_time_ms >= longest_loop_completion_time:
                longest_loop_completion_time = loop_complete_time_ms

            last_loop_complete_timestamp = get_current_timestamp_ms()


def long_short_loop(stop_event):
    with LogExceptionContext(re_raise=False):
        logger = get_logger()
        logger.info("Starting longshort loop")

        last_loop_complete_timestamp = get_current_timestamp_ms()
        last_slack_message_timestamp = 0

        while not stop_event.is_set():
            strategies = LongShortGroupQuery.get_strategies()

            current_state_dict = {
                "total_available_pairs": 0,
                "no_loan_available_err": 0,
                "total_num_symbols": 0,
                "sell_symbols": [],
                "buy_symbols": [],
                "strategies": [],
            }

            for strategy in strategies:
                update_long_short_exits(strategy, current_state_dict)

                if strategy.is_disabled is False:
                    current_state_dict["strategies"].append(strategy.name)

            strategies = LongShortGroupQuery.get_strategies()
            for strategy in strategies:
                update_long_short_enters(strategy, current_state_dict)

            if get_current_timestamp_ms() >= last_slack_message_timestamp + 60000:
                create_log(
                    msg=format_pair_trade_loop_msg(
                        current_state_dict, last_loop_complete_timestamp
                    ),
                    level=LogLevel.INFO,
                )
                last_slack_message_timestamp = get_current_timestamp_ms()

            last_loop_complete_timestamp = get_current_timestamp_ms()


class PredictionService:
    def __init__(self):
        self.stop_event = Event()
        self.rule_based_process = None
        self.long_short_process = None

    def start(self):
        if self.rule_based_process is None or not self.rule_based_process.is_alive():
            self.rule_based_process = Process(
                target=rule_based_loop, args=(self.stop_event,), daemon=False
            )
            self.long_short_process = Process(
                target=long_short_loop, args=(self.stop_event,), daemon=False
            )
            self.rule_based_process.start()
            self.long_short_process.start()

    def stop(self):
        if self.rule_based_process and self.rule_based_process.is_alive():
            self.stop_event.set()
            self.rule_based_process.join()
            self.long_short_process.join()
            self.rule_based_process = None
            self.long_short_process = None


prediction_service = PredictionService()


@app.get("/", response_class=Response)
def webserver_root():
    return Response(
        content="Curious?", media_type="text/plain", status_code=status.HTTP_200_OK
    )


def start_server():
    create_tables()
    logger = get_logger()
    if run_background_processes() is True:
        prediction_service.start()
    logger.info("Starting server")
    uvicorn.run("main:app", host="0.0.0.0", port=get_service_port(), log_level="info")


def stop_server():
    prediction_service.stop()
    os._exit(0)


if __name__ == "__main__":
    test_db_conn()
    try:
        start_server()
    finally:
        stop_server()
