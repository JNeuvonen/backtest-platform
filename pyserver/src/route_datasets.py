import os
from fastapi import APIRouter
from constants import DB_DATASETS

from db import create_connection, get_column_detailed_info, get_dataset_table


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
router = APIRouter()


@router.get("/{dataset_name}")
async def get_dataset_by_name(dataset_name: str) -> dict:
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    table_info = get_dataset_table(conn, dataset_name)
    return {"dataset": table_info}


@router.get("/{dataset_name}/col-info/{column_name}")
async def get_dataset_col_info(dataset_name: str, column_name: str) -> dict:
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    col_info = get_column_detailed_info(conn, dataset_name, column_name)
    return {"column": col_info}
