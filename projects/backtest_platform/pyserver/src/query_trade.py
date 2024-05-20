import json
from typing import Dict, List
from sqlalchemy import BigInteger, Boolean, Column, Float, ForeignKey, Integer, String
from log import LogExceptionContext

from orm import Base, Session


class Trade(Base):
    __tablename__ = "trade"
    id = Column(Integer, primary_key=True)
    open_price = Column(Float)
    close_price = Column(Float)
    open_time = Column(BigInteger)
    close_time = Column(BigInteger)
    direction = Column(String)
    net_result = Column(Float)
    percent_result = Column(Float)
    backtest_id = Column(Integer, ForeignKey("backtest.id"))
    predictions = Column(String)
    prices = Column(String)
    is_short_trade = Column(Boolean)
    dataset_name = Column(String)

    def serialize(self, prices, predictions):
        self.prices = json.dumps(prices)
        self.predictions = json.dumps(predictions)

    def deserialize(self):
        if self.prices:
            self.prices = json.loads(self.prices)
        if self.predictions:
            self.predictions = json.loads(self.predictions)
        return self


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
    def create_many(backtest_id: int, trade_data: List[dict]):
        if len(trade_data) == 0:
            return

        with LogExceptionContext():
            with Session() as session:
                trades = []
                for item in trade_data:
                    trade = Trade(
                        backtest_id=backtest_id,
                        open_price=item["open_price"],
                        close_price=item["close_price"],
                        open_time=item["open_time"],
                        close_time=item["close_time"],
                        direction=item["direction"],
                        net_result=item["net_result"],
                        percent_result=item["percent_result"],
                    )
                    trade.serialize(item.get("prices"), item.get("predictions"))
                    trades.append(trade)
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
                new_trade.serialize(trade_data["prices"], trade_data["predictions"])
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

                trades = [Trade.deserialize(item) for item in trades]
                return trades

    @staticmethod
    def update_trade(trade_id: int, updated_data: dict):
        with LogExceptionContext():
            with Session() as session:
                session.query(Trade).filter(Trade.id == trade_id).update(updated_data)
                session.commit()
