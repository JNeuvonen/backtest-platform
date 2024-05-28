from typing import Dict, List
from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Base, Session


class TradeInfoTick(Base):
    __tablename__ = "trade_info_tick"

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey("trade.id"))
    price = Column(Float, nullable=False)
    kline_open_time_ms = Column(BigInteger, nullable=False)


class TradeInfoTickQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = TradeInfoTick(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def get_trade_info_ticks_by_trade_id(trade_id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(TradeInfoTick)
                    .filter(TradeInfoTick.trade_id == trade_id)
                    .all()
                )

    @staticmethod
    def create_many(entries: List[Dict]):
        with LogExceptionContext():
            with Session() as session:
                session.bulk_insert_mappings(TradeInfoTick, entries)
                session.commit()
