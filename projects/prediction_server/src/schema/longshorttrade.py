from typing import Dict
from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer
from common_python.pred_serv_orm import Base, Session
from log import LogExceptionContext


class LongShortTrade(Base):
    __tablename__ = "long_short_trade"
    id = Column(Integer, primary_key=True)

    long_short_group_id = Column(Integer, ForeignKey("long_short_group.id"))
    long_side_trade_id = Column(Integer, ForeignKey("trade.id"))
    short_side_trade_id = Column(Integer, ForeignKey("trade.id"))

    net_result = Column(Float)
    percent_result = Column(Float)
    open_time_ms = Column(BigInteger)
    close_time_ms = Column(BigInteger)


class LongShortTradeQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                trade = LongShortTrade(**fields)
                session.add(trade)
                session.commit()
                return trade.id

    @staticmethod
    def update(trade_id, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                update_fields.pop("id", None)
                non_null_update_fields = {
                    k: v for k, v in update_fields.items() if v is not None
                }
                session.query(LongShortTrade).filter(
                    LongShortTrade.id == trade_id
                ).update(non_null_update_fields, synchronize_session="fetch")
                session.commit()
