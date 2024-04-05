from pydantic import BaseModel, validator


class BodyCreateStrategy(BaseModel):
    symbol: str
    enter_trade_code: str
    exit_trade_code: str
    fetch_datasources_code: str
    data_transformations_code: str

    priority: int
    kline_size_ms: int
    klines_left_till_autoclose: int
    minimum_time_between_trades_ms: int

    allocated_size_perc: float
    take_profit_threshold_perc: float
    stop_loss_threshold_perc: float

    is_paper_trade_mode: bool
    use_time_based_close: bool
    use_profit_based_close: bool
    use_stop_loss_based_close: bool
    use_taker_order: bool

    is_leverage_allowed: bool
    is_short_selling_strategy: bool


class BodyCreateCloudLog(BaseModel):
    message: str
    level: str

    @validator("level")
    def validate_level(cls, v):
        if v not in ("exception", "info", "system", "debug"):
            raise ValueError("Invalid log level")
        return v


class BodyCreateAccount(BaseModel):
    name: str
    max_debt_ratio: float


class BodyPutStrategy(BaseModel):
    id: int
