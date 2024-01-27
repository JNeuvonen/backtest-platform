from typing import List
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from log import LogExceptionContext

from orm import Base, Session


class Trade(Base):
    __tablename__ = "trade"
    id = Column(Integer, primary_key=True)
    start_price = Column(Float)
    end_price = Column(Float)
    start_time = Column(Integer)
    end_time = Column(Integer)
    direction = Column(String)
    net_result = Column(Float)
    percent_result = Column(Float)
    backtest_id = Column(Integer, ForeignKey("backtest.id"))


class TradeQuery:
    @staticmethod
    def create_many_trade_entry(backtest_id: int, trade_data: List[dict]):
        if len(trade_data) == 0:
            return

        with LogExceptionContext():
            with Session() as session:
                trades = [
                    Trade(
                        backtest_id=backtest_id,
                        start_price=trade["open_price"],
                        end_price=trade["end_price"],
                        start_time=trade["start_time"],
                        end_time=trade["end_time"],
                        direction=trade["direction"],
                        net_result=trade["net_result"],
                        percent_result=trade["percent_result"],
                    )
                    for trade in trade_data
                ]
                session.bulk_save_objects(trades)
                session.commit()

    @staticmethod
    def create_trade_entry(backtest_id: int, trade_data: dict):
        with LogExceptionContext():
            with Session() as session:
                new_trade = Trade(
                    backtest_id=backtest_id,
                    start_price=trade_data["start_price"],
                    end_price=trade_data["end_price"],
                    start_time=trade_data["start_time"],
                    end_time=trade_data["end_time"],
                    direction=trade_data["direction"],
                    net_result=trade_data["net_result"],
                    percent_result=trade_data["percent_result"],
                )
                session.add(new_trade)
                session.commit()
                return new_trade.id

    @staticmethod
    def fetch_trades_by_backtest_id(backtest_id: int):
        with LogExceptionContext():
            with Session() as session:
                trades = (
                    session.query(Trade).filter(Trade.backtest_id == backtest_id).all()
                )
                return trades

    @staticmethod
    def update_trade(trade_id: int, updated_data: dict):
        with LogExceptionContext():
            with Session() as session:
                session.query(Trade).filter(Trade.id == trade_id).update(updated_data)
                session.commit()
