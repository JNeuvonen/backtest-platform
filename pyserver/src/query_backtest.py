import json
from typing import Dict, List
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String

from log import LogExceptionContext
from orm import Base, Session


class Backtest(Base):
    __tablename__ = "backtest"
    id = Column(Integer, primary_key=True)
    open_long_trade_cond = Column(String)
    open_short_trade_cond = Column(String)
    close_long_trade_cond = Column(String)
    close_short_trade_cond = Column(String)
    use_time_based_close = Column(Boolean)
    use_profit_based_close = Column(Boolean)
    use_stop_loss_based_close = Column(Boolean)
    use_short_selling = Column(Boolean)
    klines_until_close = Column(Integer)
    name = Column(String)
    data = Column(String)
    trade_count = Column(Integer)
    profit_factor = Column(Float)
    gross_profit = Column(Float)
    gross_loss = Column(Float)
    model_weights_id = Column(Integer, ForeignKey("model_weights.id"))
    train_job_id = Column(Integer, ForeignKey("train_job.id"))
    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    start_balance = Column(Float)
    end_balance = Column(Float)
    result_perc = Column(Float)
    take_profit_threshold_perc = Column(Float)
    stop_loss_threshold_perc = Column(Float)
    backtest_range_start = Column(Integer)
    backtest_range_end = Column(Integer)
    best_trade_result_perc = Column(Float)
    worst_trade_result_perc = Column(Float)
    buy_and_hold_result_net = Column(Float)
    buy_and_hold_result_perc = Column(Float)
    share_of_winning_trades_perc = Column(Float)
    share_of_losing_trades_perc = Column(Float)
    max_drawdown_perc = Column(Float)
    cagr = Column(Float)
    market_exposure_time = Column(Float)
    risk_adjusted_return = Column(Float)
    buy_and_hold_cagr = Column(Float)

    def serialize_data(self, backtest_data):
        self.data = json.dumps(backtest_data)


class BacktestQuery:
    @staticmethod
    def deserialize_data(backtest: Backtest):
        if backtest and backtest.data:
            backtest.data = json.loads(backtest.data)
        return backtest

    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = Backtest(**fields)
                session.add(entry)
                session.commit()
                return entry.id

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

    @staticmethod
    def fetch_backtests_by_dataset_id(dataset_id: int):
        with LogExceptionContext():
            with Session() as session:
                backtests = (
                    session.query(Backtest)
                    .filter(Backtest.dataset_id == dataset_id)
                    .all()
                )
                return [BacktestQuery.deserialize_data(bt) for bt in backtests]

    @staticmethod
    def delete_backtests_by_ids(ids: List[int]):
        with LogExceptionContext():
            with Session() as session:
                session.query(Backtest).filter(Backtest.id.in_(ids)).delete(
                    synchronize_session=False
                )
                session.commit()
