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


class BodyCreateTrain(BaseModel):
    num_epochs: int
    save_model_after_every_epoch: bool
    backtest_on_val_set: bool
    enter_trade_criteria: str
    exit_trade_criteria: str
