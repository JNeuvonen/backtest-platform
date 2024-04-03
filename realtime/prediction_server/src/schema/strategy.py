from typing import Dict
from log import LogExceptionContext
from orm import Base, Session
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy import DateTime
from sqlalchemy.sql import func


class Strategy(Base):
    __tablename__ = "strategy"
    id = Column(Integer, primary_key=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    symbol = Column(String)
    enter_trade_code = Column(String)
    exit_trade_code = Column(String)
    fetch_datasources_code = Column(String)
    data_transformations_code = Column(String)

    priority = Column(Integer)
    kline_size_ms = Column(Integer)
    prev_kline_ms = Column(Integer)
    minimum_time_between_trades_ms = Column(Integer)
    klines_left_till_autoclose = Column(Integer)
    time_on_trade_open_ms = Column(Integer, default=0)

    price_on_trade_open = Column(Float)
    allocated_size_perc = Column(Float)
    take_profit_threshold_perc = Column(Float)
    stop_loss_threshold_perc = Column(Float)

    use_time_based_close = Column(Boolean, nullable=False)
    use_profit_based_close = Column(Boolean, nullable=False)
    use_stop_loss_based_close = Column(Boolean, nullable=False)
    use_taker_order = Column(Boolean)

    should_enter_trade = Column(Boolean, default=False)
    should_close_trade = Column(Boolean, default=False)

    is_paper_trade_mode = Column(Boolean)
    is_leverage_allowed = Column(Boolean)
    is_short_selling_strategy = Column(Boolean)
    is_disabled = Column(Boolean, default=False)
    is_in_position = Column(Boolean, default=False)


class StrategyQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = Strategy(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def get_strategies():
        with LogExceptionContext():
            with Session() as session:
                return session.query(Strategy).all()

    @staticmethod
    def update_strategy(strategy_id: int, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                session.query(Strategy).filter(Strategy.id == strategy_id).update(
                    update_fields
                )
                session.commit()
