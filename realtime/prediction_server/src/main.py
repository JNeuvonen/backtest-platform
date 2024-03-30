import uvicorn
import os
from contextlib import asynccontextmanager

from threading import Event, Thread
from orm import create_tables, test_db_conn
from config import get_service_port
from fastapi import FastAPI
from log import get_logger
from api.v1.strategy import router as v1_strategy_router
from strategy import get_trading_decisions
from schema.strategy import StrategyQuery


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    yield


class Routers:
    V1_STRATEGY = "/v1/strategy"


app = FastAPI(lifespan=lifespan)
app.include_router(v1_strategy_router, prefix=Routers.V1_STRATEGY)


class PredictionService:
    def __init__(self):
        self.stop_event = Event()
        self.thread = Thread(target=self.make_predictions_loop)

    def make_predictions_loop(self):
        logger = get_logger()
        logger.info("Initiating prediction service")

        while not self.stop_event.is_set():
            strategies = StrategyQuery.get_strategies()
            active_strategies = 0
            for strategy in strategies:
                trading_decisions = get_trading_decisions(strategy)

                print(trading_decisions)

                if not strategy.is_disabled:
                    active_strategies += 1
            logger.info(
                f"Prediction loop completed. Active strategies: {active_strategies}"
            )
            self.stop_event.wait(2)

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


def start_server():
    logger = get_logger()
    create_tables()
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