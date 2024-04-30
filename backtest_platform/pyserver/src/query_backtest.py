import json
from typing import Dict, List
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String

from log import LogExceptionContext
from orm import Base, Session
from query_backtest_statistics import BacktestStatistics, BacktestStatisticsQuery
from utils import combine_dicts


class Backtest(Base):
    __tablename__ = "backtest"
    id = Column(Integer, primary_key=True)

    model_weights_id = Column(Integer, ForeignKey("model_weights.id"))
    train_job_id = Column(Integer, ForeignKey("train_job.id"))
    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    dataset_name = Column(String)

    name = Column(String)
    candle_interval = Column(String)

    open_trade_cond = Column(String)
    close_trade_cond = Column(String)

    long_short_buy_cond = Column(String)
    long_short_sell_cond = Column(String)
    long_short_exit_cond = Column(String)

    is_short_selling_strategy = Column(Boolean)
    is_long_short_strategy = Column(Boolean)
    is_ml_based_strategy = Column(Boolean)

    use_time_based_close = Column(Boolean)
    use_profit_based_close = Column(Boolean)
    use_stop_loss_based_close = Column(Boolean)
    use_short_selling = Column(Boolean)

    klines_until_close = Column(Integer)
    backtest_range_start = Column(Integer)
    backtest_range_end = Column(Integer)


class BacktestQuery:
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
                backtest_statistics = BacktestStatisticsQuery.fetch_backtest_by_id(
                    backtest_id
                )
                return combine_dicts(
                    [backtest_data.__dict__, backtest_statistics.__dict__]
                )

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
                    session.query(Backtest, BacktestStatistics)
                    .join(
                        BacktestStatistics,
                        Backtest.id == BacktestStatistics.backtest_id,
                    )
                    .filter(Backtest.train_job_id == train_job_id)
                    .all()
                )
                combined_backtests = [
                    combine_dicts([backtest.__dict__, stats.__dict__])
                    for backtest, stats in backtests
                ]
                return combined_backtests

    @staticmethod
    def fetch_backtests_by_dataset_id(dataset_id: int):
        with LogExceptionContext():
            with Session() as session:
                backtests = (
                    session.query(Backtest, BacktestStatistics)
                    .join(
                        BacktestStatistics,
                        Backtest.id == BacktestStatistics.backtest_id,
                    )
                    .filter(Backtest.dataset_id == dataset_id)
                    .all()
                )
                combined_backtests = [
                    combine_dicts([backtest.__dict__, stats.__dict__])
                    for backtest, stats in backtests
                ]
                return combined_backtests

    @staticmethod
    def delete_backtests_by_ids(ids: List[int]):
        with LogExceptionContext():
            with Session() as session:
                session.query(Backtest).filter(Backtest.id.in_(ids)).delete(
                    synchronize_session=False
                )
                session.commit()

    @staticmethod
    def fetch_many_backtests(backtest_ids: List[int]):
        with LogExceptionContext():
            with Session() as session:
                backtests = (
                    session.query(Backtest, BacktestStatistics)
                    .join(
                        BacktestStatistics,
                        Backtest.id == BacktestStatistics.backtest_id,
                    )
                    .filter(Backtest.id.in_(backtest_ids))
                    .all()
                )
                combined_backtests = [
                    combine_dicts([backtest.__dict__, stats.__dict__])
                    for backtest, stats in backtests
                ]
                return combined_backtests

    @staticmethod
    def fetch_dataset_name_by_id(dataset_id: int):
        with LogExceptionContext():
            with Session() as session:
                dataset_name = (
                    session.query(Backtest.dataset_name)
                    .filter(Backtest.id == dataset_id)
                    .scalar()
                )
                return dataset_name

    @staticmethod
    def fetch_all_long_short_backtests():
        with LogExceptionContext():
            with Session() as session:
                backtests = (
                    session.query(Backtest, BacktestStatistics)
                    .join(
                        BacktestStatistics,
                        Backtest.id == BacktestStatistics.backtest_id,
                    )
                    .filter(Backtest.is_long_short_strategy == True)
                    .all()
                )
                print(backtests)
                combined_backtests = [
                    combine_dicts([backtest.__dict__, stats.__dict__])
                    for backtest, stats in backtests
                ]
                return combined_backtests
