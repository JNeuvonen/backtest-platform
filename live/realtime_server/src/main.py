from orm import create_tables, test_db_conn
import uvicorn
from schema.strategy import StrategyQuery

from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):  # pylint: disable=unused-argument, redefined-outer-name
    create_tables()
    StrategyQuery.create_entry({"open_long_trade_cond": "Testtt"})
    yield


app = FastAPI(lifespan=lifespan)


def start_server():
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    test_db_conn()
    start_server()
