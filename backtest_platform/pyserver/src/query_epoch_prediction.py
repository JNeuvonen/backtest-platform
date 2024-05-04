from typing import Dict, List
from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer
from log import LogExceptionContext
from orm import Base, Session


class EpochPrediction(Base):
    __tablename__ = "epoch_prediction"

    id = Column(Integer, primary_key=True)
    weights_id = Column(Integer, ForeignKey("model_weights.id"))
    kline_open_time = Column(BigInteger)
    prediction = Column(Float)


class EpochPredictionQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = EpochPrediction(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def create_many(weights_id: int, entries: List[Dict]):
        with LogExceptionContext():
            with Session() as session:
                new_entries = [
                    EpochPrediction(weights_id=weights_id, **entry) for entry in entries
                ]
                session.bulk_save_objects(new_entries)
                session.commit()
                return [entry.id for entry in new_entries]

    @staticmethod
    def create_many_from_ml_template(
        weights_id: int,
        predictions: List[List[float]],
        val_kline_open_times,
    ):
        with LogExceptionContext():
            with Session() as session:
                entries = [
                    EpochPrediction(
                        weights_id=weights_id,
                        prediction=pred[0],
                        kline_open_time=time.item(),
                    )
                    for pred, time in zip(predictions, val_kline_open_times.to_numpy())
                ]
                session.bulk_save_objects(entries)
                session.commit()

    @staticmethod
    def get_entries_by_backtest_id_sorted(backtest_id: int):
        with Session() as session:
            return (
                session.query(EpochPrediction)
                .filter_by(backtest_id=backtest_id)
                .order_by(EpochPrediction.kline_open_time.asc())
                .all()
            )
