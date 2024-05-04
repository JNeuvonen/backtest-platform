from sqlalchemy import Column, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import relationship
from log import LogExceptionContext

from orm import Base, Session
from query_epoch_prediction import EpochPredictionQuery


class ModelWeights(Base):
    __tablename__ = "model_weights"

    id = Column(Integer, primary_key=True)
    train_job_id = Column(Integer, ForeignKey("train_job.id"))
    epoch = Column(Integer)
    weights = Column(LargeBinary, nullable=False)
    train_loss = Column(Float)
    val_loss = Column(Float)


class ModelWeightsQuery:
    @staticmethod
    def create_model_weights_entry(
        train_job_id: int,
        epoch: int,
        weights: bytes,
        train_loss: float,
        val_loss: float,
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
                session.add(new_model_weight)
                session.commit()
                return new_model_weight.id

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
                return weights_data 

    @staticmethod
    def fetch_metadata_by_epoch(train_job_id: int, epoch: int):
        with LogExceptionContext():
            with Session() as session:
                metadata = (
                    session.query(
                        ModelWeights.id,
                        ModelWeights.epoch,
                        ModelWeights.train_loss,
                        ModelWeights.val_loss,
                    )
                    .filter(
                        ModelWeights.train_job_id == train_job_id,
                        ModelWeights.epoch == epoch,
                    )
                    .one()
                )
                return metadata

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
                        "val_predictions": [],
                    }
                    for weight in weights_metadata
                ]

                return weights_metadata_dict
