import uvicorn
from contextlib import asynccontextmanager

from orm import create_tables, test_db_conn
from config import get_service_port
from fastapi import FastAPI
from log import get_logger
from schema.strategy import StrategyQuery


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    create_tables()
    StrategyQuery.create_entry({"open_long_trade_cond": "Testtt"})
    yield


app = FastAPI(lifespan=lifespan)


def start_server():
    logger = get_logger()
    logger.info("Starting server")
    uvicorn.run("main:app", host="0.0.0.0", port=get_service_port(), log_level="info")


if __name__ == "__main__":
    test_db_conn()
    start_server()
