from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, validator
from datetime import datetime


class DataTransformation(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    transformation_code: str


class BodyCreateStrategy(BaseModel):
    name: str
    symbol: str
    base_asset: str
    quote_asset: str
    enter_trade_code: str
    exit_trade_code: str
    fetch_datasources_code: str
    candle_interval: str

    priority: int
    kline_size_ms: int
    maximum_klines_hold_time: int
    minimum_time_between_trades_ms: int
    num_req_klines: int
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

    data_transformations: List[DataTransformation] = Field(default_factory=list)


class BodyCreateLongShortStrategy(BaseModel):
    name: str
    candle_interval: str
    buy_cond: str
    sell_cond: str
    exit_cond: str

    num_req_klines: int
    max_simultaneous_positions: int
    kline_size_ms: int
    klines_until_close: int

    max_leverage_ratio: float
    take_profit_threshold_perc: float
    stop_loss_threshold_perc: float

    use_time_based_close: float
    use_profit_based_close: float
    use_stop_loss_based_close: float

    use_taker_order: bool

    asset_universe: List[Dict]
    data_transformations: List[DataTransformation] = Field(default_factory=list)

    @validator("name")
    def name_must_contain_symbol(cls, v):
        if "{SYMBOL}" not in v:
            raise ValueError('name must contain "{SYMBOL}"')
        return v


class BodyCreateCloudLog(BaseModel):
    message: str
    level: str
    source_program: int

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
    strategy_id: Optional[int] = None
    quantity: float
    open_price: float
    symbol: str
    direction: str

    @field_validator("direction")
    @classmethod
    def validate_level(cls, v):
        if v not in ("LONG", "SHORT"):
            raise ValueError("Invalid trade direction. Valid values are: [LONG, SHORT]")
        return v


class BodyPutStrategy(BaseModel):
    id: int
    active_trade_id: Optional[int] = None
    remaining_position_on_trade: Optional[float] = None
    quantity_on_trade_open: Optional[float] = None
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


class UpdateLongShortPairBody(BaseModel):
    id: Optional[int] = None
    long_short_group_id: Optional[int] = None
    buy_ticker_id: Optional[int] = None
    sell_ticker_id: Optional[int] = None

    buy_ticker_dataset_name: Optional[str] = None
    sell_ticker_dataset_name: Optional[str] = None

    buy_symbol: Optional[str] = None
    sell_symbol: Optional[str] = None
    buy_base_asset: Optional[str] = None
    sell_base_asset: Optional[str] = None
    buy_quote_asset: Optional[str] = None
    sell_quote_asset: Optional[str] = None

    buy_qty_precision: Optional[int] = None
    sell_qty_precision: Optional[int] = None
    buy_open_time: Optional[int] = None
    sell_open_time: Optional[int] = None

    buy_open_price: Optional[float] = None
    sell_open_price: Optional[float] = None
    buy_open_qty_in_base: Optional[float] = None
    sell_open_qty_in_quote: Optional[float] = None
    debt_open_qty_in_base: Optional[float] = None

    is_no_loan_available_err: Optional[bool] = None
    error_in_entering: Optional[bool] = None
    in_position: Optional[bool] = None
    should_close: Optional[bool] = None
    is_trade_finished: Optional[bool] = None


class Fill(BaseModel):
    price: str
    qty: str
    commission: str
    commissionAsset: str


class MarginAccountNewOrderResponseFULL(BaseModel):
    symbol: str
    orderId: int
    clientOrderId: str
    transactTime: int
    price: str
    origQty: str
    executedQty: str
    cummulativeQuoteQty: str
    status: str
    timeInForce: str
    type: str
    side: str
    marginBuyBorrowAmount: float
    marginBuyBorrowAsset: str
    isIsolated: bool
    fills: List[Fill]


class EnterLongShortPairBody(BaseModel):
    long_side_order: MarginAccountNewOrderResponseFULL
    short_side_order: MarginAccountNewOrderResponseFULL


class ExitLongShortPairBody(BaseModel):
    long_side_order: MarginAccountNewOrderResponseFULL
    short_side_order: MarginAccountNewOrderResponseFULL
