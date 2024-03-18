from typing import List, Optional
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
    enter_trade_cond: str
    exit_trade_cond: str
    price_col: str


class BodyDeleteDatasets(BaseModel):
    dataset_names: List[str]


class BodyCreateManualBacktest(BaseModel):
    open_long_trade_cond: str
    close_long_trade_cond: str
    open_short_trade_cond: str
    close_short_trade_cond: str
    use_short_selling: bool
    use_time_based_close: bool
    dataset_id: int
    name: Optional[str] = None
    klines_until_close: Optional[int] = None
    trading_fees_perc: float
    slippage_perc: float
