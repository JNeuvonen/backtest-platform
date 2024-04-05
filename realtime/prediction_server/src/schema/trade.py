from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSON
from orm import Base


class Trade(Base):
    __tablename__ = "trade"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategy.id"), nullable=False)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    open_time_ms = Column(Integer)
    close_time_ms = Column(Integer)

    open_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=True)
    net_result = Column(Float, nullable=True)
    percent_result = Column(Float, nullable=True)

    direction = Column(String, nullable=False)

    profit_history = Column(JSON, nullable=False, default=lambda: [])
