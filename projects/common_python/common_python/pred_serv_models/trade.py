from typing import Dict, List
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
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Base, Session
from sqlalchemy.orm import joinedload, relationship


class Trade(Base):
    __tablename__ = "trade"
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    strategy_id = Column(Integer, ForeignKey("strategy.id"), nullable=True)
    pair_trade_group_id = Column(Integer, ForeignKey("long_short_group.id"))
    strategy_group_id = Column(Integer, ForeignKey("strategy_group.id"))

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    open_time_ms = Column(BigInteger, nullable=False)
    close_time_ms = Column(BigInteger)

    open_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)

    close_price = Column(Float)
    net_result = Column(Float)
    percent_result = Column(Float)

    direction = Column(String, nullable=False)

    profit_history = Column(JSON, nullable=False, default=lambda: [])

    info_ticks = relationship("TradeInfoTick", backref="trade", lazy="joined")


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
    def update_trade(trade_id, update_fields: Dict, filter_nulls=False):
        with LogExceptionContext():
            with Session() as session:
                if filter_nulls is True:
                    non_null_update_fields = {
                        k: v for k, v in update_fields.items() if v is not None
                    }
                    session.query(Trade).filter(Trade.id == trade_id).update(
                        non_null_update_fields
                    )
                else:
                    session.query(Trade).filter(Trade.id == trade_id).update(
                        update_fields
                    )
                session.commit()

    @staticmethod
    def get_trade_by_id(trade_id: int):
        with LogExceptionContext():
            with Session() as session:
                return session.query(Trade).filter(Trade.id == trade_id).first()

    @staticmethod
    def get_open_trades():
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(Trade)
                    .options(joinedload(Trade.info_ticks))
                    .filter(Trade.close_price == None)
                    .filter(Trade.strategy_id != None)
                    .all()
                )

    @staticmethod
    def fetch_many_by_id(trade_ids: List[int]):
        with LogExceptionContext():
            with Session() as session:
                return session.query(Trade).filter(Trade.id.in_(trade_ids)).all()

    @staticmethod
    def fetch_by_longshort_group_id(id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(Trade).filter(Trade.pair_trade_group_id == id).all()
                )

    @staticmethod
    def update(trade_id, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                update_fields.pop("id", None)
                non_null_update_fields = {
                    k: v for k, v in update_fields.items() if v is not None
                }
                session.query(Trade).filter(Trade.id == trade_id).update(
                    non_null_update_fields, synchronize_session=False
                )
                session.commit()
