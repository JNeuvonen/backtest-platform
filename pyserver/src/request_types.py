from typing import List
from pydantic import BaseModel


class BodyModelData(BaseModel):
    name: str
    target_col: str
    drop_cols: List[str]
    null_fill_strategy: str
    model: str
    hyper_params_and_optimizer_code: str
    validation_split: List[int]


class BodyExecPython(BaseModel):
    code: str
    null_fill_strategy: str = "NONE"


class BodyUpdateTimeseriesCol(BaseModel):
    new_timeseries_col: str


class BodyUpdateDatasetName(BaseModel):
    new_dataset_name: str


class BodyRenameColumn(BaseModel):
    old_col_name: str
    new_col_name: str
