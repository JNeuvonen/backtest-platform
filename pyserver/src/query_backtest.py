import json
from sqlalchemy import Column, ForeignKey, Integer, String

from log import LogExceptionContext
from orm import Base, Session


class Backtest(Base):
    __tablename__ = "backtest"
    id = Column(Integer, primary_key=True)
    enter_and_exit_trade_criteria = Column(String)
    data = Column(String)
    model_weights_id = Column(Integer, ForeignKey("model_weights.id"))

    def serialize_data(self, backtest_data):
        self.data = json.dumps(backtest_data)


class BacktestQuery:
    @staticmethod
    def create_backtest_entry(enter_exit_criteria: str, data, model_weights_id: int):
        with LogExceptionContext():
            with Session() as session:
                new_backtest = Backtest(
                    enter_and_exit_trade_criteria=enter_exit_criteria,
                    data=json.dumps(data),
                    model_weights_id=model_weights_id,
                )
                session.add(new_backtest)
                session.commit()
                return new_backtest.id

    @staticmethod
    def fetch_backtest_by_id(backtest_id: int):
        with LogExceptionContext():
            with Session() as session:
                backtest_data = (
                    session.query(Backtest).filter(Backtest.id == backtest_id).first()
                )
                return backtest_data

    @staticmethod
    def update_backtest_data(backtest_id: int, new_data: dict):
        with LogExceptionContext():
            with Session() as session:
                session.query(Backtest).filter(Backtest.id == backtest_id).update(
                    {"data": json.dumps(new_data)}
                )
                session.commit()
