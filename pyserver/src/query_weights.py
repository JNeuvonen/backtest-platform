import json
from typing import List
from sqlalchemy import Column, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import relationship
from log import LogExceptionContext

from orm import Base, Session


class ModelWeights(Base):
    __tablename__ = "model_weights"

    id = Column(Integer, primary_key=True)
    train_job_id = Column(Integer, ForeignKey("train_job.id"))
    epoch = Column(Integer)
    weights = Column(LargeBinary, nullable=False)
    train_job = relationship("TrainJob")
    train_loss = Column(Float)
    val_loss = Column(Float)
    val_predictions = Column(String)
    train_predictions = Column(String)

    def serialize_val_predictions(self, list_preds):
        self.val_predictions = json.dumps(list_preds)

    def serialize_train_predictions(self, list_preds):
        self.train_predictions = json.dumps(list_preds)

    def deserialize(self):
        self.val_predictions = json.loads(self.val_predictions)
        return self

    @staticmethod
    def deserialize_val_predictions(val_predictions):
        return json.loads(val_predictions)


class ModelWeightsQuery:
    @staticmethod
    def create_model_weights_entry(
        train_job_id: int,
        epoch: int,
        weights: bytes,
        train_loss: float,
        val_loss: float,
        val_predictions: List[float],
        train_predictions: List[float],
    ):
        with LogExceptionContext():
            with Session() as session:
                new_model_weight = ModelWeights(
                    train_job_id=train_job_id,
                    epoch=epoch,
                    weights=weights,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                new_model_weight.serialize_val_predictions(val_predictions)
                new_model_weight.serialize_train_predictions(train_predictions)
                session.add(new_model_weight)
                session.commit()

    @staticmethod
    def fetch_model_weights_by_epoch(train_job_id: int, epoch: int):
        with LogExceptionContext():
            with Session() as session:
                weights_data = (
                    session.query(ModelWeights.weights)
                    .filter(
                        ModelWeights.train_job_id == train_job_id,
                        ModelWeights.epoch == epoch,
                    )
                    .scalar()
                )
                return weights_data.deserialize()

    @staticmethod
    def fetch_model_weights_by_train_job_id(train_job_id: int):
        with LogExceptionContext():
            with Session() as session:
                weights_metadata = (
                    session.query(
                        ModelWeights.id,
                        ModelWeights.epoch,
                        ModelWeights.train_loss,
                        ModelWeights.val_loss,
                        ModelWeights.val_predictions,
                        ModelWeights.train_predictions,
                    )
                    .filter(ModelWeights.train_job_id == train_job_id)
                    .all()
                )

                weights_metadata_dict = [
                    {
                        "id": weight.id,
                        "epoch": weight.epoch,
                        "train_loss": weight.train_loss,
                        "val_loss": weight.val_loss,
                        "val_predictions": weight.val_predictions,
                        "train_predictions": weight.train_predictions,
                    }
                    for weight in weights_metadata
                ]

                return weights_metadata_dict
