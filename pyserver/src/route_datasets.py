import os

from fastapi import APIRouter, Body, HTTPException
from constants import DB_DATASETS, DB_DATASETS_UTIL
from pydantic import BaseModel
from db import (
    create_connection,
    get_column_detailed_info,
    get_dataset_table,
    get_timeseries_col,
    rename_column,
    update_timeseries_col,
)


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
router = APIRouter()


@router.get("/{dataset_name}")
async def route_get_dataset_by_name(dataset_name: str) -> dict:
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    table_info = get_dataset_table(conn, dataset_name)
    return {"dataset": table_info}


@router.get("/{dataset_name}/col-info/{column_name}")
async def route_get_dataset_col_info(dataset_name: str, column_name: str) -> dict:
    datasets_conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    utils_conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS_UTIL))
    timeseries_col = get_timeseries_col(utils_conn, dataset_name)
    col_info = get_column_detailed_info(
        datasets_conn, dataset_name, column_name, timeseries_col
    )
    return {"column": col_info, "timeseries_col": timeseries_col}


class TimeseriesColumn(BaseModel):
    new_timeseries_col: str | None


@router.put("/{dataset_name}/update-timeseries-col")
async def route_update_timeseries_col(dataset_name: str, body: TimeseriesColumn):
    utils_conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS_UTIL))
    success = update_timeseries_col(utils_conn, dataset_name, body.new_timeseries_col)
    if success:
        return {"message": "Update successful"}
    raise HTTPException(status_code=400, detail="Update failed")


@router.get("/all-columns")
async def route_all_columns():
    return {"columns": "hello world"}


@router.post("/{dataset_name}/rename-column")
async def route_rename_column(
    dataset_name: str, old_col_name: str = Body(...), new_col_name: str = Body(...)
):
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    is_success = await rename_column(conn, dataset_name, old_col_name, new_col_name)
    if is_success:
        return {"is_success": is_success}
    raise HTTPException(status_code=400, detail="Failed to rename the column")
