from typing import List, Optional
from pydantic import BaseModel, validator

from constants import NullFillStrategy, ScalingStrategy
from datetime import datetime


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


class BodyCloneIndicators(BaseModel):
    existing_datasets: List[str]
    new_datasets: List[str]
    candle_interval: Optional[str] = None
    use_futures: Optional[bool] = None


class BodyCreateCodePreset(BaseModel):
    code: str
    category: str
    name: str


class BodyCodePreset(BaseModel):
    id: int
    code: str
    category: str
    name: str
    description: Optional[str] = None
    label: Optional[str] = None


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


class BodyCreateLongShortBacktest(BaseModel):
    backtest_data_range: List[int]
    datasets: List[str]
    data_transformations: List[int]
    sell_cond: str
    buy_cond: str
    exit_cond: str
    max_simultaneous_positions: int
    max_leverage_ratio: float
    candle_interval: str
    fetch_latest_data: bool
    use_time_based_close: bool
    use_profit_based_close: bool
    use_stop_loss_based_close: bool
    trading_fees_perc: float
    slippage_perc: float
    short_fee_hourly: float
    take_profit_threshold_perc: float
    stop_loss_threshold_perc: float
    name: Optional[str] = None
    klines_until_close: Optional[int] = None


class BodyRuleBasedOnUniverse(BaseModel):
    name: Optional[str] = None
    candle_interval: str
    datasets: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    data_transformations: List[int]
    klines_until_close: Optional[int] = None
    open_trade_cond: str
    close_trade_cond: str
    fetch_latest_data: bool
    is_cryptocurrency_datasets: bool
    is_short_selling_strategy: bool
    use_time_based_close: bool
    use_profit_based_close: bool
    use_stop_loss_based_close: bool
    take_profit_threshold_perc: float
    stop_loss_threshold_perc: float
    short_fee_hourly: float
    trading_fees_perc: float
    slippage_perc: float
    allocation_per_symbol: float

    @validator("start_date", "end_date", pre=True)
    def parse_date(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value


class BodyMLBasedBacktest(BaseModel):
    backtest_data_range: List[int]
    fetch_latest_data: bool
    dataset_name: str
    id_of_model: int
    train_run_id: int
    epoch: int
    enter_long_trade_cond: str
    exit_long_trade_cond: str
    enter_short_trade_cond: str
    exit_short_trade_cond: str
    allow_shorts: bool
    use_time_based_close: bool
    use_profit_based_close: bool
    use_stop_loss_based_close: bool
    trading_fees_perc: float
    slippage_perc: float
    short_fee_hourly: float
    take_profit_threshold_perc: float
    stop_loss_threshold_perc: float
    name: Optional[str] = None
    klines_until_close: Optional[int] = None


class BodyDeleteManyBacktestsById:
    list_of_ids: List[int]


class BodyCreateMassBacktest(BaseModel):
    crypto_symbols: List[str]
    stock_market_symbols: List[str]
    original_backtest_id: int
    fetch_latest_data: bool


class BodyCreateDataTransformation(BaseModel):
    transformation_code: str
    name: str
