from typing import Dict, List
from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer, String
from log import LogExceptionContext
from orm import Base, Session


class BacktestHistoryTick(Base):
    __tablename__ = "backtest_history"

    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest.id"))

    kline_open_time = Column(BigInteger)
    portfolio_worth = Column(Float)
    buy_and_hold_worth = Column(Float)
    prediction = Column(Float)
    position = Column(Float)
    short_debt = Column(Float)
    cash = Column(Float)
    price = Column(Float)


class BacktestHistoryQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = BacktestHistoryTick(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def get_entry_by_id(entry_id: int):
        with Session() as session:
            return session.query(BacktestHistoryTick).filter_by(id=entry_id).first()

    @staticmethod
    def delete_entry(entry_id: int):
        with Session() as session:
            entry = session.query(BacktestHistoryTick).filter_by(id=entry_id).first()
            if entry:
                session.delete(entry)
                session.commit()

    @staticmethod
    def create_many(backtest_id: int, entries: List[Dict]):
        with LogExceptionContext():
            with Session() as session:
                new_entries = [
                    BacktestHistoryTick(backtest_id=backtest_id, **entry)
                    for entry in entries
                ]
                session.bulk_save_objects(new_entries)
                session.commit()
                return [entry.id for entry in new_entries]

    @staticmethod
    def get_entries_by_backtest_id_sorted(backtest_id: int):
        with Session() as session:
            return (
                session.query(BacktestHistoryTick)
                .filter_by(backtest_id=backtest_id)
                .order_by(BacktestHistoryTick.kline_open_time.asc())
                .all()
            )

    @staticmethod
    def get_entries_by_backtest_id_sorted_partial(backtest_id: int):
        with Session() as session:
            return [
                {
                    "kline_open_time": entry.kline_open_time,
                    "portfolio_worth": entry.portfolio_worth,
                }
                for entry in session.query(
                    BacktestHistoryTick.kline_open_time,
                    BacktestHistoryTick.portfolio_worth,
                )
                .filter_by(backtest_id=backtest_id)
                .order_by(BacktestHistoryTick.kline_open_time.asc())
                .all()
            ]

    @staticmethod
    def fetch_many(backtest_ids: List[int]):
        with Session() as session:
            return [
                session.query(BacktestHistoryTick)
                .filter(BacktestHistoryTick.backtest_id == backtest_id)
                .order_by(BacktestHistoryTick.kline_open_time.asc())
                .all()
                for backtest_id in backtest_ids
            ]

    @staticmethod
    def get_first_last_kline_times_by_backtest_id(backtest_id: int):
        with Session() as session:
            first_entry = (
                session.query(BacktestHistoryTick.kline_open_time)
                .filter_by(backtest_id=backtest_id)
                .order_by(BacktestHistoryTick.kline_open_time.asc())
                .first()
            )
            last_entry = (
                session.query(BacktestHistoryTick.kline_open_time)
                .filter_by(backtest_id=backtest_id)
                .order_by(BacktestHistoryTick.kline_open_time.desc())
                .first()
            )
            return {
                "first_kline_open_time": first_entry.kline_open_time
                if first_entry
                else None,
                "last_kline_open_time": last_entry.kline_open_time
                if last_entry
                else None,
            }

    @staticmethod
    def get_ticks_by_kline_open_times(backtest_id: int, kline_open_times: List[int]):
        with Session() as session:
            return [
                {
                    "backtest_id": entry.backtest_id,
                    "kline_open_time": entry.kline_open_time,
                    "portfolio_worth": entry.portfolio_worth,
                }
                for entry in session.query(
                    BacktestHistoryTick.backtest_id,
                    BacktestHistoryTick.kline_open_time,
                    BacktestHistoryTick.portfolio_worth,
                )
                .filter(
                    BacktestHistoryTick.backtest_id == backtest_id,
                    BacktestHistoryTick.kline_open_time.in_(kline_open_times),
                )
                .order_by(BacktestHistoryTick.kline_open_time.asc())
                .all()
            ]
