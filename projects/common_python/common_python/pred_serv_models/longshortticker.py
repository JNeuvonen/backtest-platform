from typing import Dict
from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Integer, String
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Base, Session


class LongShortTicker(Base):
    __tablename__ = "long_short_ticker"
    id = Column(Integer, primary_key=True)
    long_short_group_id = Column(Integer, ForeignKey("long_short_group.id"))

    symbol = Column(String)
    base_asset = Column(String)
    quote_asset = Column(String)
    dataset_name = Column(String, unique=True)

    last_kline_open_time_sec = Column(BigInteger, default=None)
    trade_quantity_precision = Column(Integer)

    is_valid_buy = Column(Boolean, default=False)
    is_valid_sell = Column(Boolean, default=False)
    is_on_pred_serv_err = Column(Boolean, default=False)


class LongShortTickerQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                ticker = LongShortTicker(**fields)
                session.add(ticker)
                session.commit()
                return ticker.id

    @staticmethod
    def update(ticker_id, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                update_fields.pop("id", None)
                non_null_update_fields = {
                    k: v for k, v in update_fields.items() if v is not None
                }
                session.query(LongShortTicker).filter(
                    LongShortTicker.id == ticker_id
                ).update(non_null_update_fields, synchronize_session="fetch")
                session.commit()

    @staticmethod
    def get_all_by_group_id(group_id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(LongShortTicker)
                    .filter(LongShortTicker.long_short_group_id == group_id)
                    .all()
                )

    @staticmethod
    def get_all():
        with LogExceptionContext():
            with Session() as session:
                return session.query(LongShortTicker).all()
