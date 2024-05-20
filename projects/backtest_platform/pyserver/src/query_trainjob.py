import json
from typing import List
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from constants import AppConstants
from log import LogExceptionContext
from orm import Base, Session
from query_ml_validation_set_prices import MLValidationSetPriceQuery
from query_weights import ModelWeightsQuery
from query_model import Model, ModelQuery
from request_types import BodyCreateTrain
from query_dataset import Dataset, DatasetQuery
from dataset import read_all_cols_matching_kline_open_times, read_columns_to_mem


class TrainJob(Base):
    __tablename__ = "train_job"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("model.id"))

    is_training = Column(Boolean)
    name = Column(String)
    num_epochs = Column(Integer)
    epochs_ran = Column(Integer, default=0, nullable=False)
    save_model_every_epoch = Column(Boolean)
    backtest_on_validation_set = Column(Boolean)
    device = Column(String)


class TrainJobQuery:
    @classmethod
    def get_train_job_detailed(cls, train_job_id: int):
        train_job: TrainJob = cls.get_train_job(train_job_id)
        model: Model = ModelQuery.fetch_model_by_id(train_job.model_id)
        dataset: Dataset = DatasetQuery.fetch_dataset_by_id(model.dataset_id)
        epochs = ModelWeightsQuery.fetch_model_weights_by_train_job_id(train_job_id)

        return {
            "train_job": train_job,
            "model": model,
            "dataset_metadata": dataset,
            "epochs": epochs,
            "validation_set_ticks": MLValidationSetPriceQuery.fetch_all_by_job_id(
                train_job_id
            ),
        }

    @staticmethod
    def create_train_job(model_id: int, request_body: BodyCreateTrain):
        with LogExceptionContext():
            with Session() as session:
                new_train_job = TrainJob(
                    model_id=model_id,
                    num_epochs=request_body.num_epochs,
                    save_model_every_epoch=request_body.save_model_after_every_epoch,
                    backtest_on_validation_set=request_body.backtest_on_val_set,
                    is_training=True,
                )
                session.add(new_train_job)
                session.commit()
                return new_train_job.id

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
    def fetch_train_jobs_by_model(model_id):
        with LogExceptionContext():
            with Session() as session:
                train_jobs = (
                    session.query(TrainJob).filter(TrainJob.model_id == model_id).all()
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
    def fetch_all_metadata_by_model_id(cls, model_id: int):
        with LogExceptionContext():
            with Session() as session:
                train_jobs: List[TrainJob] = (
                    session.query(TrainJob).filter(TrainJob.model_id == model_id).all()
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
                for item in train_jobs:
                    item.is_training = False
                session.commit()

    @staticmethod
    def add_device(train_job_id: int, device: str):
        with Session() as session:
            train_job = (
                session.query(TrainJob).filter(TrainJob.id == train_job_id).first()
            )
            if train_job:
                train_job.device = device
                session.commit()
                return True
            else:
                return False
