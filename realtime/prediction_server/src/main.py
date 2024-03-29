import uvicorn
from contextlib import asynccontextmanager

from orm import create_tables, test_db_conn
from config import get_service_port
from fastapi import FastAPI
from log import get_logger
from api.v1.strategy import router as v1_strategy_router


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    create_tables()
    yield


class Routers:
    V1_STRATEGY = "/v1/strategy"


app = FastAPI(lifespan=lifespan)
app.include_router(v1_strategy_router, prefix=Routers.V1_STRATEGY)


def start_server():
    logger = get_logger()
    logger.info("Starting server")
    uvicorn.run("main:app", host="0.0.0.0", port=get_service_port(), log_level="info")


if __name__ == "__main__":
    test_db_conn()
    start_server()
