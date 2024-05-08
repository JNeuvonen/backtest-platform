import pickle

from sqlalchemy import Column, ForeignKey, Integer, LargeBinary
from log import LogExceptionContext
from orm import Base, Session


class ModelScaler(Base):
    __tablename__ = "model_scaler"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("model.id"))

    scaler_blob = Column(LargeBinary)


class ModelScalerQuery:
    @staticmethod
    def create_scaler_entry(model_id: int, scaler_blob: bytes):
        with LogExceptionContext():
            with Session() as session:
                model_scaler = ModelScaler(model_id=model_id, scaler_blob=scaler_blob)
                session.add(model_scaler)
                session.commit()
                return model_scaler.id

    @staticmethod
    def get_scaler_by_model_id(model_id: int):
        with LogExceptionContext():
            with Session() as session:
                model_scaler = (
                    session.query(ModelScaler)
                    .filter(ModelScaler.model_id == model_id)
                    .one_or_none()
                )
                if model_scaler:
                    return pickle.loads(model_scaler.scaler_blob)
                return None

    @staticmethod
    def delete_scaler_by_model_id(model_id: int):
        with LogExceptionContext():
            with Session() as session:
                session.query(ModelScaler).filter(
                    ModelScaler.model_id == model_id
                ).delete()
                session.commit()
