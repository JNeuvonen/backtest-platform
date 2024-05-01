from typing import Dict
from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer, String
from log import LogExceptionContext
from orm import Base, Session


class PairTrade(Base):
    __tablename__ = "pair_trade"
    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest.id"))
    buy_trade_id = Column(Integer, ForeignKey("trade.id"))
    sell_trade_id = Column(Integer, ForeignKey("trade.id"))

    gross_result = Column(Float)
    percent_result = Column(Float)

    open_time = Column(BigInteger)
    close_time = Column(BigInteger)

    history = Column(String)


class PairTradeQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = PairTrade(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def fetch_all_by_backtest_id(backtest_id: int):
        with Session() as session:
            return (
                session.query(PairTrade)
                .filter(PairTrade.backtest_id == backtest_id)
                .all()
            )
