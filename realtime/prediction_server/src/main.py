import uvicorn
import time
from contextlib import asynccontextmanager

from orm import create_tables, test_db_conn
from config import get_service_port
from fastapi import FastAPI
from log import get_logger
from api.v1.strategy import router as v1_strategy_router
from utils import run_in_thread
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


def make_predictions_loop():
    logger = get_logger()
    logger.info("Initiating prediction service")

    while True:
        strategies = StrategyQuery.get_strategies()
        for item in strategies:
            print(item)
        logger.info("Hello from pred loop")
        time.sleep(5)


def start_server():
    logger = get_logger()
    create_tables()
    run_in_thread(make_predictions_loop)
    logger.info("Starting server")
    uvicorn.run("main:app", host="0.0.0.0", port=get_service_port(), log_level="info")


if __name__ == "__main__":
    test_db_conn()
    start_server()
