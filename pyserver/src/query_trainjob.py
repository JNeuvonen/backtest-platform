import json
from typing import List
from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import relationship
from log import LogExceptionContext
from orm import Base, Session
from query_weights import ModelWeightsQuery
from query_model import Model, ModelQuery
from request_types import BodyCreateTrain
from query_dataset import Dataset, DatasetQuery


class TrainJob(Base):
    __tablename__ = "train_job"

    id = Column(Integer, primary_key=True)
    is_training = Column(Boolean)
    name = Column(String)
    model_name = Column(String)
    num_epochs = Column(Integer)
    epochs_ran = Column(Integer, default=0, nullable=False)
    save_model_every_epoch = Column(Boolean)
    backtest_on_validation_set = Column(Boolean)
    backtest_kline_open_times = Column(String)
    backtest_prices = Column(String)

    model_weights = relationship("ModelWeights", overlaps="train_job")

    def serialize_kline_open_times(self, kline_open_times):
        self.backtest_kline_open_times = json.dumps(kline_open_times)

    def serialize_prices(self, prices):
        self.backtest_prices = json.dumps(prices)


class TrainJobQuery:
    @classmethod
    def get_train_job_detailed(cls, train_job_id: int):
        train_job: TrainJob = cls.get_train_job(train_job_id)
        model: Model = ModelQuery.fetch_model_by_name(train_job.model_name)
        dataset: Dataset = DatasetQuery.fetch_dataset_by_id(model.dataset_id)
        epochs = ModelWeightsQuery.fetch_model_weights_by_train_job_id(train_job_id)
        return {
            "train_job": train_job,
            "model": model,
            "dataset": dataset,
            "epochs": epochs,
        }

    @staticmethod
    def create_train_job(model_name: str, request_body: BodyCreateTrain):
        with LogExceptionContext():
            with Session() as session:
                new_train_job = TrainJob(
                    model_name=model_name,
                    num_epochs=request_body.num_epochs,
                    save_model_every_epoch=request_body.save_model_after_every_epoch,
                    backtest_on_validation_set=request_body.backtest_on_val_set,
                    is_training=True,
                )
                session.add(new_train_job)
                session.commit()
                return new_train_job.id

    @staticmethod
    def set_backtest_data(
        train_job_id: int,
        prices: List[float],
        kline_open_times: List[float],
    ):
        with Session() as session:
            query = session.query(TrainJob)
            train_job: TrainJob = query.filter(
                getattr(TrainJob, "id") == train_job_id
            ).first()
            train_job.serialize_kline_open_times(kline_open_times)
            train_job.serialize_prices(prices)
            session.commit()

    @staticmethod
    def set_curr_epoch(train_job_id: int, epoch: int):
        with Session() as session:
            session.query(TrainJob).filter(TrainJob.id == train_job_id).update(
                {"epochs_ran": epoch}
            )
            session.commit()

    @staticmethod
    def is_job_training(id: int):
        with LogExceptionContext():
            with Session() as session:
                query = session.query(TrainJob)
                train_job = query.filter(getattr(TrainJob, "id") == id).first()

                return train_job.is_training

    @staticmethod
    def get_train_job(value, field="id"):
        with LogExceptionContext():
            with Session() as session:
                query = session.query(TrainJob)
                train_job_data = query.filter(getattr(TrainJob, field) == value).first()
                return train_job_data

    @staticmethod
    def fetch_train_jobs_by_model(model_name):
        with LogExceptionContext():
            with Session() as session:
                train_jobs = (
                    session.query(TrainJob)
                    .filter(TrainJob.model_name == model_name)
                    .all()
                )
                return train_jobs

    @staticmethod
    def set_training_status(train_job_id: int, is_training: bool):
        with LogExceptionContext():
            with Session() as session:
                train_job = (
                    session.query(TrainJob).filter(TrainJob.id == train_job_id).first()
                )
                if train_job:
                    train_job.is_training = is_training
                    session.commit()
                    return True
                else:
                    return False

    @classmethod
    def fetch_all_metadata_by_name(cls, model_name: str):
        with LogExceptionContext():
            with Session() as session:
                train_jobs: List[TrainJob] = (
                    session.query(TrainJob)
                    .filter(TrainJob.model_name == model_name)
                    .all()
                )

                ret = []
                for item in train_jobs:
                    ret_item = {
                        "train": item,
                        "weights": ModelWeightsQuery.fetch_model_weights_by_train_job_id(
                            item.id
                        ),
                    }
                    ret.append(ret_item)
                return ret

    @classmethod
    def on_shutdown_cleanup(cls):
        with LogExceptionContext():
            with Session() as session:
                train_jobs = (
                    session.query(TrainJob).filter(TrainJob.is_training == True).all()
                )
                print(train_jobs)
                for item in train_jobs:
                    item.is_training = False
                session.commit()
