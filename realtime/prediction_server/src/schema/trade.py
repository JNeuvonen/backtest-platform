from typing import Dict
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSON
from orm import Base, Session
from log import LogExceptionContext


class Trade(Base):
    __tablename__ = "trade"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategy.id"), nullable=False)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    open_time_ms = Column(BigInteger, nullable=False)
    close_time_ms = Column(BigInteger)

    open_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=True)
    net_result = Column(Float, nullable=True)
    percent_result = Column(Float, nullable=True)

    direction = Column(String, nullable=False)

    profit_history = Column(JSON, nullable=False, default=lambda: [])


class TradeQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = Trade(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def get_trades():
        with LogExceptionContext():
            with Session() as session:
                return session.query(Trade).all()

    @staticmethod
    def update_trade(trade_id: int, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                session.query(Trade).filter(Trade.id == trade_id).update(update_fields)
                session.commit()
