from pydantic import BaseModel


class BodyCreateStrategy(BaseModel):
    symbol: str
    enter_trade_code: str
    exit_trade_code: str
    fetch_datasources_code: str
    data_transformations_code: str

    priority: int
    kline_size_ms: int
    klines_left_till_autoclose: int

    allocated_size_perc: float
    take_profit_threshold_perc: float
    stop_loss_threshold_perc: float

    use_testnet: bool
    use_time_based_close: bool
    use_profit_based_close: bool
    use_stop_loss_based_close: bool
    use_taker_order: bool

    is_leverage_allowed: bool
    is_short_selling_strategy: bool
    is_disabled: bool
    is_in_position: bool
