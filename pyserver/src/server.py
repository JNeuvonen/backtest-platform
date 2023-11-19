import logging
import sqlite3
import uvicorn
import os
from fastapi import FastAPI, Response, status, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from db import (
    create_connection,
    get_column_names,
    get_tables,
    insert_binance_scrape_job,
    poll_scrape_jobs,
)
from pydantic import BaseModel
from bina_load_data import ScrapeJob
from typing import Optional, List
from constants import DB_DATASETS, DB_WORKER_QUEUE, LOG_FILE
from routes_binance import router as binance_router
import threading
from log import Logger

app = FastAPI()
app.include_router(binance_router, prefix="/binance")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
logger = Logger(
    os.path.join(APP_DATA_PATH if APP_DATA_PATH is not None else "", LOG_FILE),
    logging.INFO,
)


def get_path(relative_to_app_data: str):
    return os.path.join(APP_DATA_PATH, relative_to_app_data)


class ScrapeJobModel(BaseModel):
    url: str
    target_dataset: Optional[bool] = False
    columns_to_drop: Optional[List[str]] = None


@app.get("/", response_class=Response)
def read_root():
    return Response(
        content="Hello World", media_type="text/plain", status_code=status.HTTP_200_OK
    )


@app.get("/tables")
def read_tables():
    db_path = get_path(DB_DATASETS)
    try:
        db = create_connection(db_path)
        tables = get_tables(db)
        db.close()
        return {"tables": tables}
    except sqlite3.Error as e:
        error_detail = {"error": str(e), "db_path": db_path}
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/create_scrape_job")
def create_scrape_job(scrape_job: ScrapeJobModel):
    job = ScrapeJob(
        url=scrape_job.url,
        target_dataset=False
        if scrape_job.target_dataset is None
        else scrape_job.target_dataset,
        columns_to_drop=scrape_job.columns_to_drop,
    )

    if job.path is None:
        return {"message": "invalid URL provided"}

    db_worker_queue_path = get_path(DB_WORKER_QUEUE)
    worker_queue_conn = create_connection(db_worker_queue_path)
    insert_success = insert_binance_scrape_job(worker_queue_conn, job)

    if insert_success:
        return {"message": "ScrapeJob created successfully", "scrape_job": job}

    return Response(
        content="ScrapeJob was not created succesfully",
        media_type="text/plain",
        status_code=status.HTTP_400_BAD_REQUEST,
    )


@app.post("/poll_scrape_jobs")
def poll_scrape_jobs_endpoint():
    logger.info("Started worker thread for polling scrape jobs.")
    worker_thread = threading.Thread(
        target=poll_scrape_jobs, args=(APP_DATA_PATH, logger)
    )
    worker_thread.start()

    return Response(
        content="Initiated worker queue poll",
        media_type="text/plain",
        status_code=status.HTTP_200_OK,
    )


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
        raise HTTPException(status_code=500, detail=error_detail)

    db_connection.close()
    return {"table": table_name, "columns": columns}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
