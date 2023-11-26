import os
from fastapi import APIRouter
from constants import DB_DATASETS

from db import create_connection, get_dataset_table


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
router = APIRouter()


@router.get("/{dataset_name}")
async def get_binance_exchange_info(dataset_name: str) -> dict:
    conn = create_connection(os.path.join(APP_DATA_PATH, DB_DATASETS))
    table_info = get_dataset_table(conn, dataset_name)
    return {"dataset": table_info}
