from typing import List
from sqlalchemy import (
    Float,
    LargeBinary,
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    ForeignKey,
    update,
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy import create_engine
from config import append_app_data_path
from constants import DATASET_UTILS_DB_PATH
from log import LogExceptionContext
from request_types import BodyCreateTrain, BodyModelData

DATABASE_URI = f"sqlite:///{append_app_data_path(DATASET_UTILS_DB_PATH)}"

engine = create_engine(DATABASE_URI)
Base = declarative_base()
Session = sessionmaker(bind=engine)


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True)
    dataset_name = Column(String)
    timeseries_column = Column(String)


class Model(Base):
    __tablename__ = "model"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    target_col = Column(String)
    drop_cols = Column(String)
    null_fill_strategy = Column(String)
    model_code = Column(String)
    model_name = Column(String)
    optimizer_and_criterion_code = Column(String)
    validation_split = Column(String)

    dataset = relationship("Dataset")


class ModelWeights(Base):
    __tablename__ = "model_weights"

    id = Column(Integer, primary_key=True)
    train_job_id = Column(Integer, ForeignKey("train_job.id"))
    epoch = Column(Integer)
    weights = Column(LargeBinary, nullable=False)
    train_job = relationship("TrainJob")
    train_loss = Column(Float)
    val_loss = Column(Float)


class TrainJob(Base):
    __tablename__ = "train_job"

    id = Column(Integer, primary_key=True)
    is_training = Column(Boolean)
    name = Column(String)
    model_name = Column(String)
    num_epochs = Column(Integer)
    curr_epoch = Column(Integer, default=0, nullable=False)
    save_model_every_epoch = Column(Boolean)
    backtest_on_validation_set = Column(Boolean)

    model_weights = relationship("ModelWeights", overlaps="train_job")


def db_delete_all_data():
    with Session() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


def drop_tables():
    Base.metadata.drop_all(engine)


def create_tables():
    Base.metadata.create_all(engine)


def generate_name_for_train_job(model_name: str):
    train_jobs = TrainJobQuery.fetch_train_jobs_by_model(model_name)
    return f"{len(train_jobs)}: {model_name}"


class DatasetQuery:
    @staticmethod
    def update_timeseries_col(dataset_name: str, new_timeseries_col: str):
        with LogExceptionContext():
            with Session() as session:
                with LogExceptionContext():
                    session.execute(
                        update(Dataset)
                        .where(Dataset.dataset_name == dataset_name)
                        .values(timeseries_column=new_timeseries_col)
                    )
                    session.commit()

    @staticmethod
    def get_timeseries_col(dataset_name: str):
        with LogExceptionContext():
            with Session() as session:
                result = (
                    session.query(Dataset.timeseries_column)
                    .filter(Dataset.dataset_name == dataset_name)
                    .scalar()
                )
                return result

    @staticmethod
    def fetch_dataset_by_id(dataset_id: int):
        with LogExceptionContext():
            with Session() as session:
                dataset_data = (
                    session.query(Dataset).filter(Dataset.id == dataset_id).first()
                )
                return dataset_data

    @staticmethod
    def update_dataset_name(old_name: str, new_name: str):
        with LogExceptionContext():
            with Session() as session:
                session.execute(
                    update(Dataset)
                    .where(Dataset.dataset_name == old_name)
                    .values(dataset_name=new_name)
                )
                session.commit()

    @staticmethod
    def create_dataset_entry(dataset_name: str, timeseries_column: str):
        with LogExceptionContext():
            with Session() as session:
                new_dataset = Dataset(
                    dataset_name=dataset_name, timeseries_column=timeseries_column
                )
                session.add(new_dataset)
                session.commit()

    @classmethod
    def fetch_dataset_id_by_name(cls, dataset_name: str):
        with LogExceptionContext():
            with Session() as session:
                result = (
                    session.query(Dataset.id)
                    .filter(Dataset.dataset_name == dataset_name)
                    .scalar()
                )
                return result


class ModelQuery:
    @staticmethod
    def create_model_entry(dataset_id: int, model_data: BodyModelData):
        with LogExceptionContext():
            with Session() as session:
                new_model = Model(
                    dataset_id=dataset_id,
                    target_col=model_data.target_col,
                    drop_cols=",".join(model_data.drop_cols),
                    null_fill_strategy=model_data.null_fill_strategy.value,
                    model_code=model_data.model,
                    model_name=model_data.name,
                    optimizer_and_criterion_code=model_data.hyper_params_and_optimizer_code,
                    validation_split=",".join(map(str, model_data.validation_split)),
                )
                session.add(new_model)
                session.commit()

    @staticmethod
    def fetch_model_by_id(model_id: int):
        with LogExceptionContext():
            with Session() as session:
                model_data = session.query(Model).filter(Model.id == model_id).first()
                return model_data

    @staticmethod
    def fetch_models_by_dataset_id(dataset_id: int):
        with LogExceptionContext():
            with Session() as session:
                models = (
                    session.query(Model).filter(Model.dataset_id == dataset_id).all()
                )
                return models

    @staticmethod
    def fetch_model_by_name(model_name: str):
        with LogExceptionContext():
            with Session() as session:
                model_data = (
                    session.query(Model).filter(Model.model_name == model_name).first()
                )
                return model_data


class TrainJobQuery:
    @staticmethod
    def create_train_job(model_name: str, request_body: BodyCreateTrain):
        with LogExceptionContext():
            train_job_name = generate_name_for_train_job(model_name)
            with Session() as session:
                new_train_job = TrainJob(
                    name=train_job_name,
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
    def set_curr_epoch(train_job_id: int, epoch: int):
        with Session() as session:
            session.query(TrainJob).filter(TrainJob.id == train_job_id).update(
                {"curr_epoch": epoch}
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
    def fetch_model_weights_by_train_job_id(train_job_id: int):
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
                }
                for weight in weights_metadata
            ]

            return weights_metadata_dict
