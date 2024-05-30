from typing import Dict
from sqlalchemy import BigInteger, Boolean, Column, Float, ForeignKey, Integer, String
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Base, Session


class LongShortPair(Base):
    __tablename__ = "long_short_pair"

    id = Column(Integer, primary_key=True)

    long_short_group_id = Column(Integer, ForeignKey("long_short_group.id"))
    buy_ticker_id = Column(Integer, ForeignKey("long_short_ticker.id"))
    sell_ticker_id = Column(Integer, ForeignKey("long_short_ticker.id"))
    buy_side_trade_id = Column(Integer, ForeignKey("trade.id"))
    sell_side_trade_id = Column(Integer, ForeignKey("trade.id"))

    buy_ticker_dataset_name = Column(String)
    sell_ticker_dataset_name = Column(String)

    buy_symbol = Column(String)
    sell_symbol = Column(String)
    buy_base_asset = Column(String)
    sell_base_asset = Column(String)
    buy_quote_asset = Column(String)
    sell_quote_asset = Column(String)

    buy_qty_precision = Column(Integer)
    sell_qty_precision = Column(Integer)
    buy_open_time_ms = Column(BigInteger)
    sell_open_time_ms = Column(BigInteger)
    last_loan_attempt_fail_time_ms = Column(BigInteger)

    buy_open_price = Column(Float)
    sell_open_price = Column(Float)
    buy_open_qty_in_base = Column(Float)
    buy_open_qty_in_quote = Column(Float)
    sell_open_qty_in_quote = Column(Float)
    debt_open_qty_in_base = Column(Float)

    is_no_loan_available_err = Column(Boolean, default=False)
    error_in_entering = Column(Boolean, default=False)
    in_position = Column(Boolean, default=False)
    should_close = Column(Boolean, default=False)
    is_trade_finished = Column(Boolean, default=False)


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

    @staticmethod
    def get_pair_by_id(pair_id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(LongShortPair)
                    .filter(LongShortPair.id == pair_id)
                    .first()
                )

    @staticmethod
    def delete_entry(pair_id: int):
        with LogExceptionContext():
            with Session() as session:
                pair = (
                    session.query(LongShortPair)
                    .filter(LongShortPair.id == pair_id)
                    .first()
                )
                if pair:
                    session.delete(pair)
                    session.commit()

    @staticmethod
    def count_pairs_in_position():
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(LongShortPair)
                    .filter(LongShortPair.in_position == True)
                    .count()
                )

    @staticmethod
    def get_all():
        with LogExceptionContext():
            with Session() as session:
                return session.query(LongShortPair).all()
