import os
from contextlib import asynccontextmanager
from code_preset_utils import generate_default_code_presets
from utils import on_shutdown_cleanup
import uvicorn
import query_epoch_prediction
import query_ml_validation_set_prices

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from log import LogExceptionContext, get_logger
from route_binance import router as binance_router
from route_datasets import router as datasets_router
from route_model import router as model_router
from route_backtest import router as backtest_router
from route_code_preset import router as code_preset_router
from route_data_transformation import router as data_transformation_router
from streams import router as streams_router
from orm import create_tables


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):  # pylint: disable=unused-argument, redefined-outer-name
    """The code before the yield statement will be executed on boot. The code after the yield statement will be executed as a cleanup on application close."""

    with LogExceptionContext("Succesfully initiated tables"):
        create_tables()
        generate_default_code_presets()
    yield


class Routers:
    BINANCE = "/binance"
    STREAMS = "/streams"
    DATASET = "/dataset"
    MODEL = "/model"
    BACKTEST = "/backtest"
    CODE_PRESET = "/code-preset"
    DATA_TRANSFORMATION = "/data-transformation"


app = FastAPI(lifespan=lifespan)
app.include_router(binance_router, prefix=Routers.BINANCE)
app.include_router(streams_router, prefix=Routers.STREAMS)
app.include_router(datasets_router, prefix=Routers.DATASET)
app.include_router(model_router, prefix=Routers.MODEL)
app.include_router(backtest_router, prefix=Routers.BACKTEST)
app.include_router(code_preset_router, prefix=Routers.CODE_PRESET)
app.include_router(data_transformation_router, prefix=Routers.DATA_TRANSFORMATION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DATA_PATH = os.getenv("APP_DATA_PATH", "tests")


@app.post("/shutdown")
def shutdown_server():
    on_shutdown_cleanup()
    logger = get_logger()
    logger.info("Application is shutting down")
    os._exit(0)


@app.get("/frontend-dev-test-error")
async def test_error_endpoint():
    raise HTTPException(status_code=400, detail="Custom error message for testing")


@app.get("/", response_class=Response)
def read_root():
    return Response(
        content="Hello World", media_type="text/plain", status_code=status.HTTP_200_OK
    )


def run():
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    run()
