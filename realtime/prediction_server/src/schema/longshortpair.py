from typing import Dict
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from orm import Base, Session
from log import LogExceptionContext


class LongShortPair(Base):
    __tablename__ = "long_short_pair"

    id = Column(Integer, primary_key=True)

    long_short_group_id = Column(Integer, ForeignKey("long_short_group.id"))
    buy_ticker_id = Column(Integer, ForeignKey("long_short_ticker.id"))
    sell_ticker_id = Column(Integer, ForeignKey("long_short_ticker.id"))

    buy_ticker_dataset_name = Column(String)
    sell_ticker_dataset_name = Column(String)

    buy_open_time = Column(Integer)
    sell_open_time = Column(Integer)

    buy_open_price = Column(Float)
    sell_open_price = Column(Float)
    buy_open_qty_in_base = Column(Float)
    sell_open_qty_in_quote = Column(Float)
    debt_open_qty_in_base = Column(Float)

    in_position = Column(Boolean, default=False)
    should_close = Column(Boolean, default=False)


class LongShortPairQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                pair = LongShortPair(**fields)
                session.add(pair)
                session.commit()
                return pair.id

    @staticmethod
    def get_pairs_by_group_id(group_id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(LongShortPair)
                    .filter(LongShortPair.long_short_group_id == group_id)
                    .all()
                )

    @staticmethod
    def update_entry(pair_id, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                update_fields.pop("id", None)
                non_null_update_fields = {
                    k: v for k, v in update_fields.items() if v is not None
                }
                session.query(LongShortPair).filter(LongShortPair.id == pair_id).update(
                    non_null_update_fields, synchronize_session="fetch"
                )
                session.commit()
