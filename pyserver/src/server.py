import os
import sqlite3
from contextlib import asynccontextmanager
import uvicorn

from fastapi import FastAPI, Response, status, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from db import (
    DatasetUtils,
    create_connection,
    exec_sql,
    get_column_names,
    get_tables,
)
from constants import DB_DATASETS
from route_binance import router as binance_router
from route_datasets import router as datasets_router
from sql_statements import CREATE_DATASET_UTILS_TABLE
from streams import router as streams_router


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):  # pylint: disable=unused-argument, redefined-outer-name
    exec_sql(DatasetUtils.get_path(), CREATE_DATASET_UTILS_TABLE)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(binance_router, prefix="/binance")
app.include_router(streams_router, prefix="/streams")
app.include_router(datasets_router, prefix="/dataset")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")


def get_path(relative_to_app_data: str):
    return os.path.join(APP_DATA_PATH, relative_to_app_data)


@app.get("/", response_class=Response)
def read_root():
    return Response(
        content="Hello World", media_type="text/plain", status_code=status.HTTP_200_OK
    )


@app.get("/tables")
async def read_tables():
    db_path = get_path(DB_DATASETS)
    try:
        db = create_connection(db_path)
        tables = get_tables(db)
        db.close()
        return {"tables": tables}
    except sqlite3.Error as e:
        error_detail = {"error": str(e), "db_path": db_path}
        raise HTTPException(status_code=500, detail=error_detail) from e


@app.get("/tables/{table_name}/columns")
def read_table_columns(table_name: str = Path(..., title="The name of the table")):
    db_path = os.path.join(APP_DATA_PATH, DB_DATASETS)
    db_connection = create_connection(db_path)

    try:
        # Get the list of column names from the table
        columns = get_column_names(db_connection, table_name)
    except sqlite3.Error as e:
        db_connection.close()
        error_detail = {
            "error": str(e),
            "app_data_path": APP_DATA_PATH,
            "db_path": db_path,
        }
        raise HTTPException(status_code=500, detail=error_detail) from e

    db_connection.close()
    return {"table": table_name, "columns": columns}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
