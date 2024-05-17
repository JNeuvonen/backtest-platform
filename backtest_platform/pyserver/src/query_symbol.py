from typing import Dict, List
from sqlalchemy import Column, ForeignKey, Integer, String
from log import LogExceptionContext
from orm import Base, Session


class Symbol(Base):
    __tablename__ = "symbol"

    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest.id"))
    symbol = Column(String)


class SymbolQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            try:
                with Session() as session:
                    symbol_entry = Symbol(**fields)
                    session.add(symbol_entry)
                    session.commit()
                    return symbol_entry.id
            except Exception as e:
                return str(e)

    @staticmethod
    def get_symbols_by_backtest(backtest_id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(Symbol)
                    .filter(Symbol.backtest_id == backtest_id)
                    .order_by(Symbol.id)
                    .all()
                )

    @staticmethod
    def create_many(backtest_id: int, fields_list: List[Dict]):
        with LogExceptionContext():
            with Session() as session:
                symbols = [
                    Symbol(backtest_id=backtest_id, **fields) for fields in fields_list
                ]
                session.add_all(symbols)
                session.commit()
                return [symbol.id for symbol in symbols]
