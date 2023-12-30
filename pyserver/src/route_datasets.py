import logging
import os


from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from constants import DB_DATASETS
from db import (
    DatasetUtils,
    create_connection,
    get_column_detailed_info,
    get_dataset_table,
    rename_column,
    rename_table,
)
from log import get_logger


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
    timeseries_col = DatasetUtils.get_timeseries_col(dataset_name)
    col_info = get_column_detailed_info(
        datasets_conn, dataset_name, column_name, timeseries_col
    )
    return {"column": col_info, "timeseries_col": timeseries_col}


class BodyUpdateTimeseriesCol(BaseModel):
    new_timeseries_col: str | None


@router.put("/{dataset_name}/update-timeseries-col")
async def route_update_timeseries_col(dataset_name: str, body: BodyUpdateTimeseriesCol):
    success = DatasetUtils.update_timeseries_col(dataset_name, body.new_timeseries_col)
    if success:
        return {"message": "Update successful"}
    raise HTTPException(status_code=400, detail="Update failed")


class BodyUpdateDatasetName(BaseModel):
    new_dataset_name: str | None


@router.put("/{dataset_name}/update-dataset-name")
async def route_update_dataset_name(dataset_name: str, body: BodyUpdateDatasetName):
    db_datasets_path = os.path.join(APP_DATA_PATH, DB_DATASETS)
    rename_success = rename_table(db_datasets_path, dataset_name, body.new_dataset_name)
    update_utils = DatasetUtils.update_dataset_name(dataset_name, body.new_dataset_name)

    if rename_success is False or update_utils is False:
        raise HTTPException(status_code=400, detail="Update failed")

    return {"message": "Update successful"}


@router.get("/all-columns")
async def route_all_columns():
    return {"columns": "hello world"}


class BodyRenameColumn(BaseModel):
    old_col_name: str
    new_col_name: str
    is_timeseries_col: bool


@router.post("/{dataset_name}/rename-column")
async def route_rename_column(dataset_name: str, body: BodyRenameColumn):
    rename_is_success = rename_column(
        os.path.join(APP_DATA_PATH, DB_DATASETS),
        dataset_name,
        body.old_col_name,
        body.new_col_name,
    )

    timeseries_col_update_success = True

    if body.is_timeseries_col is True:
        timeseries_col_update_success = DatasetUtils.update_timeseries_col(
            dataset_name, body.new_col_name
        )

    if rename_is_success and timeseries_col_update_success:
        logger = get_logger()
        await logger.log(
            f"Renamed column on table: {dataset_name} from {body.old_col_name} to {body.new_col_name}",
            logging.INFO,
        )
        return {"message": "OK"}
    raise HTTPException(status_code=400, detail="Failed to rename the column")
