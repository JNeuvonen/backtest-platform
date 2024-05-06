from typing import Dict
from sqlalchemy import Column, Integer, String
from backtest_platform.pyserver.src.log import LogExceptionContext
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
    symbol = Column(String)
    interval = Column(String)


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
