from typing import Dict
from orm import Base, Session
from sqlalchemy import Column, Integer, String


class Strategy(Base):
    __tablename__ = "strategy"
    id = Column(Integer, primary_key=True)

    open_long_trade_cond = Column(String)


class StrategyQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with Session() as session:
            entry = Strategy(**fields)
            session.add(entry)
            session.commit()
            return entry.id
