from typing import Dict
from sqlalchemy import Column, ForeignKey, Integer
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Base, Session


class RefetchStrategySignal(Base):
    __tablename__ = "refetch_strategy_signal"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategy.id"))


class RefetchStrategySignalQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = RefetchStrategySignal(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def delete_entry_by_id(entry_id: int):
        with LogExceptionContext():
            with Session() as session:
                entry = session.query(RefetchStrategySignal).get(entry_id)
                if entry:
                    session.delete(entry)
                    session.commit()
                    return True
                return False

    @staticmethod
    def get_all():
        with LogExceptionContext():
            with Session() as session:
                entries = session.query(RefetchStrategySignal).all()
                return entries

    @staticmethod
    def delete_entries_by_ids(entry_ids: list):
        with LogExceptionContext():
            with Session() as session:
                entries = (
                    session.query(RefetchStrategySignal)
                    .filter(RefetchStrategySignal.id.in_(entry_ids))
                    .all()
                )
                if entries:
                    for entry in entries:
                        session.delete(entry)
                    session.commit()
                    return True
                return False
