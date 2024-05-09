import os
import asyncio

from typing import List

from fastapi.responses import FileResponse
from config import append_app_data_path
from constants import STREAMING_DEFAULT_CHUNK_SIZE, AppConstants, NullFillStrategy
from context import HttpResponseContext
from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, status
from pydantic import BaseModel
from dataset import df_fill_nulls_on_all_cols, read_dataset_to_mem
from db import (
    add_columns_to_table,
    create_copy,
    delete_dataset_cols,
    exec_python,
    get_all_tables_and_columns,
    get_column_detailed_info,
    get_dataset_pagination,
    get_dataset_table,
    get_dataset_tables,
    get_linear_regression_analysis,
    remove_table,
    rename_column,
    rename_table,
)
from query_dataset import Dataset, DatasetBody, DatasetQuery
from query_data_transformation import DataTransformationQuery
from query_model import ModelQuery
from request_types import (
    BodyDeleteDatasets,
    BodyExecPython,
    BodyModelData,
    BodyRenameColumn,
    BodyUpdateDatasetName,
    BodyUpdateTimeseriesCol,
)
from utils import (
    PythonCode,
    add_to_datasets_db,
    read_file_to_dataframe,
    remove_all_csv_files,
)


APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
router = APIRouter()


class RoutePaths:
    TABLES = "/tables"
    GET_DATASET_BY_NAME = "/{dataset_name}"
    DATASET = "/{dataset_name}"
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
    CREATE_MODEL = "/{dataset_name}/models/create"
    FETCH_MODELS = "/{dataset_name}/models/"
    SET_TARGET_COLUMN = "/{dataset_name}/target-column"
    COPY = "/{dataset_name}/copy"
    UPDATE_PRICE_COLUMN = "/{dataset_name}/price-column"
    GET_DATASET_ROW_PAGINATION = "/{dataset_name}/pagination/{page}/{page_size}"
    DOWNLOAD = "/{dataset_name}/download"


@router.post(RoutePaths.EXEC_PYTHON_ON_COL)
async def route_exec_python_on_column(
    dataset_name: str, column_name: str, body: BodyExecPython
):
    with HttpResponseContext():
        python_program = PythonCode.on_column(dataset_name, column_name, body.code)
        exec_python(python_program)
        return {"message": "OK"}


@router.post(RoutePaths.EXEC_PYTHON_ON_DATASET)
async def route_exec_python_on_dataset(
    dataset_name: str,
    body: BodyExecPython,
):
    with HttpResponseContext():
        dataset = DatasetQuery.fetch_dataset_by_name(dataset_name)

        if dataset is None:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset was not found with name {dataset_name}",
            )

        null_fill_strat = NullFillStrategy[body.null_fill_strategy]
        python_program = PythonCode.on_dataset(dataset_name, body.code)
        exec_python(python_program)

        id = DataTransformationQuery.create_entry(
            {"transformation_code": body.code, "dataset_id": dataset.id}
        )
        df_fill_nulls_on_all_cols(dataset_name, null_fill_strat)
        return Response(
            content=str(id), status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.get(RoutePaths.TABLES)
async def route_all_tables():
    with HttpResponseContext():
        return {"tables": get_dataset_tables()}


@router.get(RoutePaths.GET_DATASET_BY_NAME)
async def route_get_dataset_by_name(dataset_name: str) -> dict:
    with HttpResponseContext():
        return {"dataset": get_dataset_table(dataset_name)}


@router.get(RoutePaths.DATASET)
async def route_update_dataset(dataset_name: str, body: DatasetBody):
    with HttpResponseContext():
        DatasetQuery.update_dataset(dataset_name, body)
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.get(RoutePaths.GET_DATASET_COL_INFO)
async def route_get_dataset_col_info(dataset_name: str, column_name: str) -> dict:
    with HttpResponseContext():
        timeseries_col = DatasetQuery.get_timeseries_col(dataset_name)
        price_col = DatasetQuery.get_price_col(dataset_name)
        target_col = DatasetQuery.get_target_col(dataset_name)

        col_info = get_column_detailed_info(
            dataset_name, column_name, timeseries_col, target_col, price_col
        )
        analysis_dict = get_linear_regression_analysis(
            dataset_name, column_name, target_col
        )

        return {
            "column": col_info,
            "timeseries_col": timeseries_col,
            "price_col": price_col,
            **analysis_dict,
        }


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


@router.put(RoutePaths.UPDATE_TIMESERIES_COL)
async def route_update_timeseries_col(dataset_name: str, body: BodyUpdateTimeseriesCol):
    with HttpResponseContext():
        DatasetQuery.update_timeseries_col(dataset_name, body.new_timeseries_col)
        return {"message": "OK"}


@router.put(RoutePaths.UPDATE_DATASET_NAME)
async def route_update_dataset_name(dataset_name: str, body: BodyUpdateDatasetName):
    with HttpResponseContext():
        rename_table(AppConstants.DB_DATASETS, dataset_name, body.new_dataset_name)
        DatasetQuery.update_dataset_name(dataset_name, body.new_dataset_name)
        return {"message": "OK"}


@router.get(RoutePaths.ROOT)
async def route_all_columns():
    with HttpResponseContext():
        return {"table_col_map": get_all_tables_and_columns(AppConstants.DB_DATASETS)}


@router.post(RoutePaths.RENAME_COLUMN)
async def route_rename_column(dataset_name: str, body: BodyRenameColumn):
    with HttpResponseContext(
        f"Renamed column on table: {dataset_name} from {body.old_col_name} to {body.new_col_name}"
    ):
        if body.old_col_name == DatasetQuery.get_timeseries_col(dataset_name):
            DatasetQuery.update_timeseries_col(dataset_name, body.new_col_name)

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
        DatasetQuery.create_dataset_entry(dataset_name, timeseries_col)
        return {"message": "OK", "shape": df.shape}


@router.post(RoutePaths.CREATE_MODEL)
async def route_create_model(dataset_name: str, body: BodyModelData):
    with HttpResponseContext():
        dataset = DatasetQuery.fetch_dataset_by_name(dataset_name)
        if dataset is None:
            raise HTTPException(
                detail=f"No dataset found for {dataset_name}", status_code=400
            )

        id = ModelQuery.create_model_entry(dataset, body)

        return Response(
            content=str(id),
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )


@router.get(RoutePaths.FETCH_MODELS)
async def route_fetch_models(dataset_name: str):
    with HttpResponseContext():
        dataset_id = DatasetQuery.fetch_dataset_id_by_name(dataset_name)
        if dataset_id is None:
            raise HTTPException(
                detail=f"No ID found for {dataset_name}", status_code=400
            )
        models = ModelQuery.fetch_models_by_dataset_id(dataset_id)
        return {"data": models}


@router.put(RoutePaths.SET_TARGET_COLUMN)
async def route_post_target_col(dataset_name: str, target_column: str):
    with HttpResponseContext():
        DatasetQuery.update_target_column(dataset_name, target_column)
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.put(RoutePaths.UPDATE_PRICE_COLUMN)
async def route_update_price_col(dataset_name: str, price_column: str):
    with HttpResponseContext():
        DatasetQuery.update_price_column(dataset_name, price_column)
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.post(RoutePaths.COPY)
async def route_copy_dataset(dataset_name: str, new_dataset_name: str):
    with HttpResponseContext():
        create_copy(dataset_name, new_dataset_name)
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.get(RoutePaths.GET_DATASET_ROW_PAGINATION)
async def route_get_dataset_pagination(dataset_name: str, page: int, page_size: int):
    return {"data": get_dataset_pagination(dataset_name, page, page_size)}


@router.get(RoutePaths.DOWNLOAD)
async def route_download_dataset(dataset_name: str):
    with HttpResponseContext():
        remove_all_csv_files(append_app_data_path(""))
        df = read_dataset_to_mem(dataset_name)
        csv_path = append_app_data_path(f"{dataset_name}.csv")
        df.to_csv(csv_path, index=False)
        return FileResponse(path=csv_path, filename=csv_path, media_type="text/csv")


@router.delete(RoutePaths.TABLES)
async def route_del_tables(body: BodyDeleteDatasets):
    with HttpResponseContext():
        datasets = body.dataset_names

        for item in datasets:
            DatasetQuery.delete_entry_by_dataset_name(item)
            remove_table(item)

        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )
