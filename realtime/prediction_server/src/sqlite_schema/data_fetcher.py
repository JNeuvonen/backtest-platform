from typing import Dict
from sqlalchemy import BigInteger, Boolean, Column, Integer, String
from log import LogExceptionContext
from sqlite_orm import Session
from sqlite_orm import Base


class DataFetcher(Base):
    __tablename__ = "data_fetcher"

    id = Column(Integer, primary_key=True)

    strategy_id = Column(Integer)
    strategy_name = Column(String, unique=True)
    fetch_datasources_code = Column(String, nullable=False)
    num_required_klines = Column(Integer, nullable=False)
    kline_size_ms = Column(Integer, nullable=False)
    last_kline_open_time_sec = Column(BigInteger, nullable=True)
    symbol = Column(String)
    interval = Column(String)
    prev_should_open_trade = Column(Boolean, default=False)
    prev_should_close_trade = Column(Boolean, default=False)
    is_on_pred_serv_err = Column(Boolean, default=False)


class DataFetcherQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = DataFetcher(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def get_by_strat_name(strategy_name: str):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(DataFetcher)
                    .filter(DataFetcher.strategy_name == strategy_name)
                    .first()
                )

    @staticmethod
    def get_by_strat_id(strat_id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(DataFetcher)
                    .filter(DataFetcher.strategy_id == strat_id)
                    .first()
                )

    @staticmethod
    def update_last_kline_open_time(strategy_id: int, new_last_kline_open_time: int):
        with LogExceptionContext():
            with Session() as session:
                entry = (
                    session.query(DataFetcher)
                    .filter(DataFetcher.strategy_id == strategy_id)
                    .first()
                )
                if entry:
                    entry.last_kline_open_time_sec = new_last_kline_open_time
                    session.commit()
                    return True
                return False

    @staticmethod
    def update_trade_flags(
        strategy_id: int,
        should_open_trade: bool,
        should_close_trade: bool,
        is_on_pred_serv_err: bool,
    ):
        with LogExceptionContext():
            with Session() as session:
                entry = (
                    session.query(DataFetcher)
                    .filter(DataFetcher.strategy_id == strategy_id)
                    .first()
                )
                if entry:
                    entry.prev_should_open_trade = should_open_trade
                    entry.prev_should_close_trade = should_close_trade
                    entry.is_on_pred_serv_err = is_on_pred_serv_err
                    session.commit()
                    return True
                return False
