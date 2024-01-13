import os
import asyncio

from typing import List
from constants import AppConstants
from context import HttpResponseContext
from fastapi import APIRouter
from pydantic import BaseModel
from db import (
    DatasetUtils,
    add_columns_to_table,
    create_connection,
    get_all_tables_and_columns,
    get_column_detailed_info,
    get_dataset_table,
    get_tables,
    rename_column,
    rename_table,
)


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
router = APIRouter()


@router.get("/tables")
async def route_all_tables():
    with HttpResponseContext():
        db_path = AppConstants.DB_DATASETS
        db = create_connection(db_path)
        tables = get_tables(db)
        db.close()
        return {"tables": tables}


@router.get("/{dataset_name}")
async def route_get_dataset_by_name(dataset_name: str) -> dict:
    with HttpResponseContext():
        conn = create_connection(AppConstants.DB_DATASETS)
        table_info = get_dataset_table(conn, dataset_name)
        return {"dataset": table_info}


@router.get("/{dataset_name}/col-info/{column_name}")
async def route_get_dataset_col_info(dataset_name: str, column_name: str) -> dict:
    with HttpResponseContext():
        datasets_conn = create_connection(AppConstants.DB_DATASETS)
        timeseries_col = DatasetUtils.get_timeseries_col(dataset_name)
        col_info = get_column_detailed_info(
            datasets_conn, dataset_name, column_name, timeseries_col
        )
        return {"column": col_info, "timeseries_col": timeseries_col}


class ColumnsToDataset(BaseModel):
    table_name: str
    columns: List[str]


@router.post("/{dataset_name}/add-columns")
async def route_post_dataset_add_columns(
    dataset_name: str, payload: List[ColumnsToDataset]
):
    with HttpResponseContext():
        asyncio.create_task(
            add_columns_to_table(AppConstants.DB_DATASETS, dataset_name, payload)
        )
        return {"message": "OK"}


class BodyUpdateTimeseriesCol(BaseModel):
    new_timeseries_col: str | None


@router.put("/{dataset_name}/update-timeseries-col")
async def route_update_timeseries_col(dataset_name: str, body: BodyUpdateTimeseriesCol):
    with HttpResponseContext():
        DatasetUtils.update_timeseries_col(dataset_name, body.new_timeseries_col)
        return {"message": "OK"}


class BodyUpdateDatasetName(BaseModel):
    new_dataset_name: str | None


@router.put("/{dataset_name}/update-dataset-name")
async def route_update_dataset_name(dataset_name: str, body: BodyUpdateDatasetName):
    with HttpResponseContext():
        rename_table(AppConstants.DB_DATASETS, dataset_name, body.new_dataset_name)
        DatasetUtils.update_dataset_name(dataset_name, body.new_dataset_name)
        return {"message": "OK"}


@router.get("/")
async def route_all_columns():
    with HttpResponseContext():
        return {"table_col_map": get_all_tables_and_columns(AppConstants.DB_DATASETS)}


class BodyRenameColumn(BaseModel):
    old_col_name: str
    new_col_name: str
    is_timeseries_col: bool


@router.post("/{dataset_name}/rename-column")
async def route_rename_column(dataset_name: str, body: BodyRenameColumn):
    with HttpResponseContext(
        f"Renamed column on table: {dataset_name} from {body.old_col_name} to {body.new_col_name}"
    ):
        rename_column(
            AppConstants.DB_DATASETS,
            dataset_name,
            body.old_col_name,
            body.new_col_name,
        )
        if body.is_timeseries_col:
            DatasetUtils.update_timeseries_col(dataset_name, body.new_col_name)

        return {"message": "OK"}
