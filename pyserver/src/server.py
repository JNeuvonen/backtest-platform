import os
from contextlib import asynccontextmanager
import uvicorn

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from log import LogExceptionContext
from route_binance import router as binance_router
from route_datasets import router as datasets_router
from route_model import router as model_router
from streams import router as streams_router
from orm import create_tables


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):  # pylint: disable=unused-argument, redefined-outer-name
    """The code before the yield statement will be executed on boot. The code after the yield statement will be executed as a cleanup on application close."""

    with LogExceptionContext("Succesfully initiated tables"):
        create_tables()
    yield


class Routers:
    BINANCE = "/binance"
    STREAMS = "/streams"
    DATASET = "/dataset"
    MODEL = "/model"


app = FastAPI(lifespan=lifespan)
app.include_router(binance_router, prefix=Routers.BINANCE)
app.include_router(streams_router, prefix=Routers.STREAMS)
app.include_router(datasets_router, prefix=Routers.DATASET)
app.include_router(model_router, prefix=Routers.MODEL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "tests")


@app.get("/", response_class=Response)
def read_root():
    return Response(
        content="Hello World", media_type="text/plain", status_code=status.HTTP_200_OK
    )


def run():
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    run()
