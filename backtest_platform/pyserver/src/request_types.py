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


class BodyCreateCodePreset(BaseModel):
    code: str
    category: str
    name: str


class BodyCreateManualBacktest(BaseModel):
    backtest_data_range: List[int]
    open_trade_cond: str
    close_trade_cond: str
    is_short_selling_strategy: bool
    use_time_based_close: bool
    use_profit_based_close: bool
    use_stop_loss_based_close: bool
    dataset_id: int
    trading_fees_perc: float
    slippage_perc: float
    short_fee_hourly: float
    take_profit_threshold_perc: float
    stop_loss_threshold_perc: float
    name: Optional[str] = None
    klines_until_close: Optional[int] = None


class BodyDeleteManyBacktestsById:
    list_of_ids: List[int]
