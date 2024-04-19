import json
from typing import Dict
from sqlalchemy import Column, ForeignKey, Integer, String
from log import LogExceptionContext
from orm import Base, Session


class MassBacktest(Base):
    __tablename__ = "mass_backtest"

    id = Column(Integer, primary_key=True)
    original_backtest_id = Column(Integer, ForeignKey("backtest.id"))

    name = Column(String)
    backtest_ids = Column(String)

    def serialize(self):
        self.backtest_ids = json.dumps(self.backtest_ids)

    def deserialize(self):
        self.backtest_ids = json.loads(self.backtest_ids)


class MassBacktestQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                fields["backtest_ids"] = []
                mass_backtest = MassBacktest(**fields)
                mass_backtest.serialize()
                session.add(mass_backtest)
                session.commit()
                return mass_backtest.id

    @staticmethod
    def get_mass_backtests():
        with LogExceptionContext():
            with Session() as session:
                mass_backtests = session.query(MassBacktest).all()
                for backtest in mass_backtests:
                    backtest.deserialize()
                return mass_backtests

    @staticmethod
    def remove_mass_backtest_by_id(id: int):
        with LogExceptionContext():
            with Session() as session:
                session.query(MassBacktest).filter(MassBacktest.id == id).delete()
                session.commit()

    @staticmethod
    def get_mass_backtest_by_original_id(original_id: int):
        with LogExceptionContext():
            with Session() as session:
                mass_backtests = (
                    session.query(MassBacktest)
                    .filter(MassBacktest.original_backtest_id == original_id)
                    .order_by(MassBacktest.id)
                    .all()
                )

                for item in mass_backtests:
                    item.deserialize()

                return mass_backtests

    @staticmethod
    def add_backtest_id(mass_backtest_id: int, new_backtest_id: int):
        with LogExceptionContext():
            with Session() as session:
                mass_backtest = (
                    session.query(MassBacktest)
                    .filter(MassBacktest.id == mass_backtest_id)
                    .first()
                )
                if mass_backtest:
                    current_ids = json.loads(mass_backtest.backtest_ids)
                    current_ids.append(new_backtest_id)
                    mass_backtest.backtest_ids = json.dumps(current_ids)
                    session.commit()
