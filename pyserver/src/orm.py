from typing import List
from pandas.io.parquet import json
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
    val_predictions = Column(String)

    def serialize_val_predictions(self, list_preds):
        self.val_predictions = json.dumps(list_preds)


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
    validation_target_before_scale = Column(String)
    validation_kline_open_times = Column(String)

    model_weights = relationship("ModelWeights", overlaps="train_job")

    def serialize_target_before_scale(self, val_targets_before_scale):
        self.validation_target_before_scale = json.dumps(val_targets_before_scale)

    def serialize_kline_open_times(self, kline_open_times):
        self.validation_kline_open_times = json.dumps(kline_open_times)


class Backtest(Base):
    __tablename__ = "backtest"
    id = Column(Integer, primary_key=True)
    enter_and_exit_trade_criteria = Column(String)
    data = Column(String)
    model_weights_id = Column(Integer, ForeignKey("model_weights.id"))

    def serialize_data(self, backtest_data):
        self.data = json.dumps(backtest_data)


class Trade(Base):
    __tablename__ = "trade"
    id = Column(Integer, primary_key=True)
    start_price = Column(Float)
    end_price = Column(Float)
    start_time = Column(Integer)
    end_time = Column(Integer)
    direction = Column(String)
    net_result = Column(Float)
    percent_result = Column(Float)
    backtest_id = Column(Integer, ForeignKey("backtest.id"))


def db_delete_all_data():
    with Session() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


def drop_tables():
    Base.metadata.drop_all(engine)


def create_tables():
    Base.metadata.create_all(engine)


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
    def set_klines_and_price_before_scale(
        train_job_id: int,
        target_before_scaling: List[float],
        kline_open_times: List[float],
    ):
        with Session() as session:
            query = session.query(TrainJob)
            train_job: TrainJob = query.filter(
                getattr(TrainJob, "id") == train_job_id
            ).first()
            train_job.serialize_target_before_scale(target_before_scaling)
            train_job.serialize_kline_open_times(kline_open_times)
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


class ModelWeightsQuery:
    @staticmethod
    def create_model_weights_entry(
        train_job_id: int,
        epoch: int,
        weights: bytes,
        train_loss: float,
        val_loss: float,
        val_predictions: List[float],
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
                    ModelWeights.val_predictions,
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
                }
                for weight in weights_metadata
            ]

            return weights_metadata_dict


class BacktestQuery:
    @staticmethod
    def create_backtest_entry(enter_exit_criteria: str, data, model_weights_id: int):
        with LogExceptionContext():
            with Session() as session:
                new_backtest = Backtest(
                    enter_and_exit_trade_criteria=enter_exit_criteria,
                    data=json.dumps(data),
                    model_weights_id=model_weights_id,
                )
                session.add(new_backtest)
                session.commit()
                return new_backtest.id

    @staticmethod
    def fetch_backtest_by_id(backtest_id: int):
        with LogExceptionContext():
            with Session() as session:
                backtest_data = (
                    session.query(Backtest).filter(Backtest.id == backtest_id).first()
                )
                return backtest_data

    @staticmethod
    def update_backtest_data(backtest_id: int, new_data: dict):
        with LogExceptionContext():
            with Session() as session:
                session.query(Backtest).filter(Backtest.id == backtest_id).update(
                    {"data": json.dumps(new_data)}
                )
                session.commit()


class TradeQuery:
    @staticmethod
    def create_many_trade_entry(backtest_id: int, trade_data: List[dict]):
        if len(trade_data) == 0:
            return

        with LogExceptionContext():
            with Session() as session:
                trades = [
                    Trade(
                        backtest_id=backtest_id,
                        start_price=trade["open_price"],
                        end_price=trade["end_price"],
                        start_time=trade["start_time"],
                        end_time=trade["end_time"],
                        direction=trade["direction"],
                        net_result=trade["net_result"],
                        percent_result=trade["percent_result"],
                    )
                    for trade in trade_data
                ]
                session.bulk_save_objects(trades)
                session.commit()

    @staticmethod
    def create_trade_entry(backtest_id: int, trade_data: dict):
        with LogExceptionContext():
            with Session() as session:
                new_trade = Trade(
                    backtest_id=backtest_id,
                    start_price=trade_data["start_price"],
                    end_price=trade_data["end_price"],
                    start_time=trade_data["start_time"],
                    end_time=trade_data["end_time"],
                    direction=trade_data["direction"],
                    net_result=trade_data["net_result"],
                    percent_result=trade_data["percent_result"],
                )
                session.add(new_trade)
                session.commit()
                return new_trade.id

    @staticmethod
    def fetch_trades_by_backtest_id(backtest_id: int):
        with LogExceptionContext():
            with Session() as session:
                trades = (
                    session.query(Trade).filter(Trade.backtest_id == backtest_id).all()
                )
                return trades

    @staticmethod
    def update_trade(trade_id: int, updated_data: dict):
        with LogExceptionContext():
            with Session() as session:
                session.query(Trade).filter(Trade.id == trade_id).update(updated_data)
                session.commit()
