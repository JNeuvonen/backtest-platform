import json
from sqlalchemy import Column, Float, ForeignKey, Integer, String

from log import LogExceptionContext
from orm import Base, Session


class Backtest(Base):
    __tablename__ = "backtest"
    id = Column(Integer, primary_key=True)
    enter_trade_cond = Column(String)
    exit_trade_cond = Column(String)
    data = Column(String)
    model_weights_id = Column(Integer, ForeignKey("model_weights.id"))
    train_job_id = Column(Integer, ForeignKey("train_job.id"))
    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    start_balance = Column(Float)
    end_balance = Column(Float)

    def serialize_data(self, backtest_data):
        self.data = json.dumps(backtest_data)


class BacktestQuery:
    @staticmethod
    def deserialize_data(backtest: Backtest):
        if backtest and backtest.data:
            backtest.data = json.loads(backtest.data)
        return backtest

    @staticmethod
    def create_backtest_entry(
        enter_trade_cond: str,
        exit_trade_cond: str,
        data,
        model_weights_id: int,
        train_job_id: int,
        start_balance: float,
        end_balance: float,
    ):
        with LogExceptionContext():
            with Session() as session:
                new_backtest = Backtest(
                    enter_trade_cond=enter_trade_cond,
                    exit_trade_cond=exit_trade_cond,
                    data=json.dumps(data),
                    model_weights_id=model_weights_id,
                    train_job_id=train_job_id,
                    start_balance=start_balance,
                    end_balance=end_balance,
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
                return BacktestQuery.deserialize_data(backtest_data)

    @staticmethod
    def update_backtest_data(backtest_id: int, new_data: dict):
        with LogExceptionContext():
            with Session() as session:
                session.query(Backtest).filter(Backtest.id == backtest_id).update(
                    {"data": json.dumps(new_data)}
                )
                session.commit()

    @staticmethod
    def fetch_backtests_by_train_job_id(train_job_id: int):
        with LogExceptionContext():
            with Session() as session:
                backtests = (
                    session.query(Backtest)
                    .filter(Backtest.train_job_id == train_job_id)
                    .all()
                )
                return [BacktestQuery.deserialize_data(bt) for bt in backtests]
