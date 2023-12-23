import os
from fastapi import APIRouter, Body, HTTPException
from constants import DB_DATASETS

from db import (
    create_connection,
    get_column_detailed_info,
    get_dataset_table,
    rename_column,
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
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    col_info = get_column_detailed_info(conn, dataset_name, column_name)
    return {"column": col_info}


@router.get("/all-columns")
async def route_all_columns():
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    return {"columns": "hello world"}


@router.post("/{dataset_name}/rename-column")
async def route_rename_column(
    dataset_name: str, old_col_name: str = Body(...), new_col_name: str = Body(...)
):
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    is_success = rename_column(conn, dataset_name, old_col_name, new_col_name)
    if is_success:
        return {"is_success": is_success}
    else:
        raise HTTPException(status_code=400, detail="Failed to rename the column")
