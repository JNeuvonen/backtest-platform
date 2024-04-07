from typing import List, Optional, Any
from pydantic import BaseModel, field_validator, conint
from datetime import datetime


class BodyCreateStrategy(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    enter_trade_code: str
    exit_trade_code: str
    fetch_datasources_code: str
    data_transformations_code: str

    priority: int
    kline_size_ms: int
    klines_left_till_autoclose: int
    minimum_time_between_trades_ms: int
    trade_quantity_precision: int

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

    @field_validator("level")
    @classmethod
    def validate_level(cls, level):
        if level not in ("exception", "info", "system", "debug"):
            raise ValueError("Invalid log level")
        return level


class BodyCreateAccount(BaseModel):
    name: str
    max_debt_ratio: float


class BodyCreateTrade(BaseModel):
    open_time_ms: int
    strategy_id: int
    quantity: float
    open_price: float
    direction: str

    @field_validator("direction")
    @classmethod
    def validate_level(cls, v):
        if v not in ("LONG", "SHORT"):
            raise ValueError("Invalid trade direction. Valid values are: [LONG, SHORT]")
        return v


class BodyPutStrategy(BaseModel):
    id: int
    name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    symbol: Optional[str] = None
    base_asset: Optional[str] = None
    quote_asset: Optional[str] = None
    enter_trade_code: Optional[str] = None
    exit_trade_code: Optional[str] = None
    fetch_datasources_code: Optional[str] = None
    data_transformations_code: Optional[str] = None
    trade_quantity_precision: Optional[int] = None
    priority: Optional[int] = None
    kline_size_ms: Optional[int] = None
    prev_kline_ms: Optional[int] = None
    minimum_time_between_trades_ms: Optional[int] = None
    maximum_klines_hold_time: Optional[int] = None
    klines_left_till_autoclose: Optional[int] = None
    time_on_trade_open_ms: Optional[int] = None
    price_on_trade_open: Optional[float] = None
    allocated_size_perc: Optional[float] = None
    take_profit_threshold_perc: Optional[float] = None
    stop_loss_threshold_perc: Optional[float] = None
    use_time_based_close: Optional[bool] = None
    use_profit_based_close: Optional[bool] = None
    use_stop_loss_based_close: Optional[bool] = None
    use_taker_order: Optional[bool] = None
    should_enter_trade: Optional[bool] = None
    should_close_trade: Optional[bool] = None
    is_paper_trade_mode: Optional[bool] = None
    is_leverage_allowed: Optional[bool] = None
    is_short_selling_strategy: Optional[bool] = None
    is_disabled: Optional[bool] = None
    is_in_position: Optional[bool] = None


class BodyUpdateTradeClose(BaseModel):
    cumulative_quote_quantity: float
    quantity: float
    price: float
    close_time_ms: float


class BodyPutTrade(BaseModel):
    id: int
    strategy_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    open_time_ms: Optional[int] = None
    close_time_ms: Optional[int] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    net_result: Optional[float] = None
    percent_result: Optional[float] = None
    direction: Optional[str] = None
    profit_history: Optional[List[Any]] = None

    @field_validator("direction")
    def validate_direction(cls, v):
        if v not in (None, "LONG", "SHORT"):
            raise ValueError("Invalid trade direction. Valid values are: [LONG, SHORT]")
        return v
