from typing import Dict
from sqlalchemy import Column, ForeignKey, Integer, String
from log import LogExceptionContext
from orm import Base, Session


class MassBacktest(Base):
    __tablename__ = "mass_backtest"

    id = Column(Integer, primary_key=True)
    original_backtest_id = Column(Integer, ForeignKey("backtest.id"))

    name = Column(String)


class MassBacktestQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                mass_backtest = MassBacktest(**fields)
                session.add(mass_backtest)
                session.commit()
                return mass_backtest.id

    @staticmethod
    def get_mass_backtests():
        with LogExceptionContext():
            with Session() as session:
                return session.query(MassBacktest).all()

    @staticmethod
    def remove_mass_backtest_by_id(backtest_id: int):
        with LogExceptionContext():
            with Session() as session:
                session.query(MassBacktest).filter(
                    MassBacktest.id == backtest_id
                ).delete()
                session.commit()

    @staticmethod
    def get_mass_backtest_by_original_id(original_id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(MassBacktest)
                    .filter(MassBacktest.original_backtest_id == original_id)
                    .order_by(MassBacktest.id)
                    .all()
                )
