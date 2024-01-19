import os
import asyncio

from typing import List
from constants import STREAMING_DEFAULT_CHUNK_SIZE, AppConstants, NullFillStrategy
from context import HttpResponseContext
from fastapi import APIRouter, Query, UploadFile
from pydantic import BaseModel
from db import (
    DatasetUtils,
    add_columns_to_table,
    delete_dataset_cols,
    exec_python,
    get_all_tables_and_columns,
    get_column_detailed_info,
    get_dataset_table,
    get_dataset_tables,
    rename_column,
    rename_table,
)
from utils import PythonCode, add_to_datasets_db, read_file_to_dataframe


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
router = APIRouter()


class RoutePaths:
    ALL_TABLES = "/tables"
    GET_DATASET_BY_NAME = "/{dataset_name}"
    GET_DATASET_COL_INFO = "/{dataset_name}/col-info/{column_name}"
    ADD_COLUMNS = "/{dataset_name}/add-columns"
    UPDATE_TIMESERIES_COL = "/{dataset_name}/update-timeseries-col"
    UPDATE_DATASET_NAME = "/{dataset_name}/update-dataset-name"
    ROOT = "/"
    RENAME_COLUMN = "/{dataset_name}/rename-column"
    DELETE_COLUMNS = "/{dataset_name}/delete-cols"
    EXEC_PYTHON_ON_COL = "/{dataset_name}/exec-python/{column_name}"
    EXEC_PYTHON_ON_DATASET = "/{dataset_name}/exec-python"
    UPLOAD_TIMESERIES_DATA = "/upload-timeseries-dataset"


class BodyExecPython(BaseModel):
    code: str


@router.post(RoutePaths.EXEC_PYTHON_ON_COL)
async def route_exec_python_on_column(
    dataset_name: str, column_name: str, body: BodyExecPython
):
    with HttpResponseContext():
        python_program = PythonCode.on_column(dataset_name, column_name, body.code)
        exec_python(python_program)
        return {"message": "OK"}


@router.post(RoutePaths.EXEC_PYTHON_ON_DATASET)
async def route_exec_python_on_dataset(dataset_name: str, body: BodyExecPython):
    with HttpResponseContext():
        python_program = PythonCode.on_dataset(dataset_name, body.code)
        exec_python(python_program)
        return {"message": "OK"}


@router.get(RoutePaths.ALL_TABLES)
async def route_all_tables():
    with HttpResponseContext():
        return {"tables": get_dataset_tables()}


@router.get(RoutePaths.GET_DATASET_BY_NAME)
async def route_get_dataset_by_name(dataset_name: str) -> dict:
    with HttpResponseContext():
        return {"dataset": get_dataset_table(dataset_name)}


@router.get(RoutePaths.GET_DATASET_COL_INFO)
async def route_get_dataset_col_info(dataset_name: str, column_name: str) -> dict:
    with HttpResponseContext():
        timeseries_col = DatasetUtils.get_timeseries_col(dataset_name)
        col_info = get_column_detailed_info(dataset_name, column_name, timeseries_col)
        return {"column": col_info, "timeseries_col": timeseries_col}


class ColumnsToDataset(BaseModel):
    table_name: str
    columns: List[str]


@router.post(RoutePaths.ADD_COLUMNS)
async def route_dataset_add_columns(
    dataset_name: str,
    payload: List[ColumnsToDataset],
    is_test_mode: bool = False,
    null_fill_strategy: str = Query("NONE"),
):
    null_fill_strat = NullFillStrategy[null_fill_strategy]
    with HttpResponseContext():
        if is_test_mode is False:
            asyncio.create_task(
                add_columns_to_table(
                    AppConstants.DB_DATASETS, dataset_name, payload, null_fill_strat
                )
            )
            return {"message": "OK"}
        await add_columns_to_table(
            AppConstants.DB_DATASETS, dataset_name, payload, null_fill_strat
        )
        return {"messsage": "OK"}


class BodyUpdateTimeseriesCol(BaseModel):
    new_timeseries_col: str


@router.put(RoutePaths.UPDATE_TIMESERIES_COL)
async def route_update_timeseries_col(dataset_name: str, body: BodyUpdateTimeseriesCol):
    with HttpResponseContext():
        DatasetUtils.update_timeseries_col(dataset_name, body.new_timeseries_col, False)
        return {"message": "OK"}


class BodyUpdateDatasetName(BaseModel):
    new_dataset_name: str


@router.put(RoutePaths.UPDATE_DATASET_NAME)
async def route_update_dataset_name(dataset_name: str, body: BodyUpdateDatasetName):
    with HttpResponseContext():
        rename_table(AppConstants.DB_DATASETS, dataset_name, body.new_dataset_name)
        DatasetUtils.update_dataset_name(dataset_name, body.new_dataset_name)
        return {"message": "OK"}


@router.get(RoutePaths.ROOT)
async def route_all_columns():
    with HttpResponseContext():
        return {"table_col_map": get_all_tables_and_columns(AppConstants.DB_DATASETS)}


class BodyRenameColumn(BaseModel):
    old_col_name: str
    new_col_name: str


@router.post(RoutePaths.RENAME_COLUMN)
async def route_rename_column(dataset_name: str, body: BodyRenameColumn):
    with HttpResponseContext(
        f"Renamed column on table: {dataset_name} from {body.old_col_name} to {body.new_col_name}"
    ):
        if body.old_col_name == DatasetUtils.get_timeseries_col(dataset_name):
            DatasetUtils.update_timeseries_col(dataset_name, body.new_col_name, True)

        rename_column(
            AppConstants.DB_DATASETS,
            dataset_name,
            body.old_col_name,
            body.new_col_name,
        )

        return {"message": "OK"}


class PayloadDeleteColumns(BaseModel):
    cols: List[str]


@router.post(RoutePaths.DELETE_COLUMNS)
async def route_del_cols(dataset_name: str, delete_cols: PayloadDeleteColumns):
    with HttpResponseContext():
        delete_dataset_cols(dataset_name, delete_cols.cols)
        return {"message": "OK"}


@router.post(RoutePaths.UPLOAD_TIMESERIES_DATA)
async def route_upload_timeseries_data(
    file: UploadFile, dataset_name: str, timeseries_col: str
):
    with HttpResponseContext():
        df = await read_file_to_dataframe(file, STREAMING_DEFAULT_CHUNK_SIZE)
        add_to_datasets_db(df, dataset_name)
        DatasetUtils.create_db_utils_entry(dataset_name, timeseries_col)
        return {"message": "OK", "shape": df.shape}
