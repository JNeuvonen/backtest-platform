from typing import Dict, List
from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer
from log import LogExceptionContext
from orm import Base, Session


class MLValidationSetPrice(Base):
    __tablename__ = "validation_set_price"

    id = Column(Integer, primary_key=True)

    train_job_id = Column(Integer, ForeignKey("train_job.id"))
    kline_open_time = Column(BigInteger)
    price = Column(Float)


class MLValidationSetPriceQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = MLValidationSetPrice(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def create_many_from_ml_template(
        train_job_id,
        prices: List[float],
        val_kline_open_times: List[int],
    ):
        with LogExceptionContext():
            with Session() as session:
                assert len(prices) == len(val_kline_open_times)

                entries = [
                    MLValidationSetPrice(
                        train_job_id=train_job_id,
                        price=price,
                        kline_open_time=kline_open_time,
                    )
                    for price, kline_open_time in zip(prices, val_kline_open_times)
                ]
                session.bulk_save_objects(entries)
                session.commit()
