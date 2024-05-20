import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from common_python.pred_serv_orm import create_tables, test_db_conn
from analytics_server.api.v1.user import router as v1_user_router
from common_python.server_config import get_service_port


class Routers:
    V1_USER = "/v1/user"


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    yield


app = FastAPI(lifespan=lifespan)


app.include_router(v1_user_router, prefix=Routers.V1_USER)


@app.get("/", response_class=Response)
def webserver_root():
    return Response(
        content="Curious?", media_type="text/plain", status_code=status.HTTP_200_OK
    )


def start_server():
    create_tables()
    uvicorn.run("main:app", host="0.0.0.0", port=get_service_port(), log_level="info")


if __name__ == "__main__":
    test_db_conn()
    start_server()
