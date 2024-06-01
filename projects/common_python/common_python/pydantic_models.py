from typing import List, Optional
from pydantic import BaseModel


class StrategyAnalyticsServUpdate(BaseModel):
    id: int
    allocation_size_perc: Optional[float] = None
    should_close_trade: Optional[float] = None
    is_in_close_only: Optional[float] = None
    stop_processing_new_candles: Optional[float] = None


class BodyAnalyticsServiceUpdate(BaseModel):
    strategies: List[StrategyAnalyticsServUpdate]
