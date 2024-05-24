import time
import uvicorn
from contextlib import asynccontextmanager
from multiprocessing import Process, Event

from fastapi import FastAPI, Response, status
from common_python.pred_serv_orm import create_tables, test_db_conn
from analytics_server.api.v1.user import router as v1_user_router
from fastapi.middleware.cors import CORSMiddleware
from common_python.server_config import get_service_port

from analytics_server.binance.collect_data import collect_data_loop
from common_python.pred_serv_models.trade import TradeQuery
from common_python.pred_serv_models.longshortpair import LongShortPairQuery
from common_python.pred_serv_models.longshortticker import LongShortTickerQuery


class DataCollectionSservice:
    def __init__(self):
        self.stop_event = Event()
        self.balance_snapshot_loop = None

    def start(self):
        time.sleep(5)
        if self.balance_snapshot_loop is None:
            self.balance_snapshot_loop = Process(
                target=collect_data_loop, args=(self.stop_event,), daemon=False
            )
            self.balance_snapshot_loop.start()

    def stop(self):
        if self.balance_snapshot_loop:
            self.stop_event.set()

            if self.balance_snapshot_loop:
                self.balance_snapshot_loop.join()


class Routers:
    V1_USER = "/v1/user"


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(v1_user_router, prefix=Routers.V1_USER)

data_collection_service = DataCollectionSservice()


@app.get("/", response_class=Response)
def webserver_root():
    return Response(
        content="Curious?", media_type="text/plain", status_code=status.HTTP_200_OK
    )


def start_server():
    create_tables()
    data_collection_service.start()
    uvicorn.run(
        "analytics_server.main:app",
        host="0.0.0.0",
        port=get_service_port(),
        log_level="info",
    )


if __name__ == "__main__":
    test_db_conn()
    start_server()
