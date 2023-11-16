import sqlite3
import uvicorn
import os
from fastapi import FastAPI, Response, status, HTTPException, Path
from db import create_connection, get_column_names, get_tables
from pydantic import BaseModel
from load_data import ScrapeJob
from typing import Optional, List
from constants import DATASETS_DB

app = FastAPI()
APP_DATA_PATH = os.getenv("APP_DATA_PATH")


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
    db_path = os.path.join(APP_DATA_PATH, DATASETS_DB) if APP_DATA_PATH else DATASETS_DB
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
    # Instantiate a ScrapeJob object from the provided data
    job = ScrapeJob(
        url=scrape_job.url,
        target_dataset=False
        if scrape_job.target_dataset is None
        else scrape_job.target_dataset,
        columns_to_drop=scrape_job.columns_to_drop,
    )

    return {"message": "ScrapeJob created successfully", "scrape_job": job}


@app.get("/tables/{table_name}/columns")
def read_table_columns(table_name: str = Path(..., title="The name of the table")):
    if APP_DATA_PATH is None:
        raise HTTPException(
            status_code=500, detail="Application data path is not configured"
        )

    db_path = os.path.join(APP_DATA_PATH, DATASETS_DB)
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
