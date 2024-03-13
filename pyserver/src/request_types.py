from typing import List
from pydantic import BaseModel

from constants import NullFillStrategy, ScalingStrategy


class BodyModelData(BaseModel):
    name: str
    drop_cols: List[str]
    null_fill_strategy: NullFillStrategy
    model: str
    hyper_params_and_optimizer_code: str
    validation_split: List[int]
    scale_target: bool
    scaling_strategy: ScalingStrategy
    drop_cols_on_train: List[str]


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


class BodyRunBacktest(BaseModel):
    dataset_name: str
    epoch_nr: int
    enter_and_exit_criteria: str
    price_col: str


class BodyDeleteDatasets(BaseModel):
    dataset_names: List[str]
