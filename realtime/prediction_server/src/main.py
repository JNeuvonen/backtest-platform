import uvicorn
import os
from contextlib import asynccontextmanager

from threading import Event, Thread
from orm import create_tables, test_db_conn
from config import get_auto_whitelisted_ip, get_service_port
from fastapi import FastAPI, Response, status
from log import get_logger
from api.v1.strategy import router as v1_strategy_router
from api.v1.log import router as v1_cloudlogs_router
from api.v1.account import router as v1_account_router
from api.v1.trade import router as v1_trade_router
from api.v1.api_key import router as v1_api_key_router
from middleware import ValidateIPMiddleware
from constants import LogLevel
from binance_utils import cleanup_unused_disk_space, gen_trading_decisions
from utils import get_current_timestamp_ms
from strategy import (
    format_pred_loop_log_msg,
    update_strategies_state_dict,
)
from schema.strategy import StrategyQuery
from schema.whitelisted_ip import WhiteListedIPQuery
from schema.cloudlog import CloudLogQuery, create_log
from fastapi.middleware.cors import CORSMiddleware
from sqlite_orm import create_sqlite_tables


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


app = FastAPI(lifespan=lifespan)

app.include_router(v1_cloudlogs_router, prefix=Routers.V1_LOGS)
app.include_router(v1_strategy_router, prefix=Routers.V1_STRATEGY)
app.include_router(v1_account_router, prefix=Routers.V1_ACC)
app.include_router(v1_trade_router, prefix=Routers.V1_TRADE)
app.include_router(v1_api_key_router, prefix=Routers.V1_API_KEY)

app.add_middleware(ValidateIPMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataProviderService:
    def __init__(self):
        self.stop_event = Event()
        self.thread = Thread(target=self.collect_data_loop)

    def collect_data_loop(self):
        pass

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = Thread(target=self.collect_data_loop)
            self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()
            self.thread = None


class PredictionService:
    def __init__(self):
        self.stop_event = Event()
        self.thread = Thread(target=self.make_predictions_loop)

    def make_predictions_loop(self):
        logger = get_logger()
        logger.info("Initiating prediction service")

        iterations_completed = 0

        last_loop_complete_timestamp = get_current_timestamp_ms()

        while not self.stop_event.is_set():
            strategies = StrategyQuery.get_strategies()

            current_state_dict = {
                "active_strategies": 0,
                "strats_on_error": 0,
                "strats_in_pos": 0,
                "strats_wanting_to_close": 0,
                "strats_wanting_to_enter": 0,
                "strats_still": 0,
            }
            strategies_info = []

            for strategy in strategies:
                trading_decisions = gen_trading_decisions(strategy)
                update_strategies_state_dict(
                    strategy, trading_decisions, current_state_dict, strategies_info
                )

            create_log(
                msg=format_pred_loop_log_msg(
                    current_state_dict, strategies_info, last_loop_complete_timestamp
                ),
                level=LogLevel.INFO,
            )

            if iterations_completed % 1000 == 0:
                CloudLogQuery.clear_outdated_logs()
                cleanup_unused_disk_space(strategies)
                iterations_completed = 0

            iterations_completed += 1
            last_loop_complete_timestamp = get_current_timestamp_ms()

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = Thread(target=self.make_predictions_loop)
            self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()
            self.thread = None


prediction_service = PredictionService()


@app.get("/", response_class=Response)
def webserver_root():
    return Response(
        content="Curious?", media_type="text/plain", status_code=status.HTTP_200_OK
    )


def start_server():
    create_tables()
    create_sqlite_tables()
    logger = get_logger()
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
