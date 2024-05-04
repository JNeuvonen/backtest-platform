import json
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from log import LogExceptionContext
from orm import Base, Session
from query_dataset import Dataset
from request_types import BodyModelData


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
    scale_target = Column(Boolean)
    scaling_strategy = Column(Integer)
    drop_cols_on_train = Column(String)

    dataset = relationship("Dataset")


class ModelQuery:
    @staticmethod
    def create_model_entry(dataset: Dataset, model_data: BodyModelData):
        with LogExceptionContext():
            with Session() as session:
                serialized_cols = json.dumps(model_data.drop_cols_on_train)
                new_model = Model(
                    dataset_id=dataset.id,
                    target_col=dataset.target_column,
                    drop_cols=",".join(model_data.drop_cols),
                    null_fill_strategy=model_data.null_fill_strategy.value,
                    model_code=model_data.model,
                    model_name=model_data.name,
                    optimizer_and_criterion_code=model_data.hyper_params_and_optimizer_code,
                    validation_split=",".join(map(str, model_data.validation_split)),
                    scale_target=model_data.scale_target,
                    scaling_strategy=model_data.scaling_strategy.value,
                    drop_cols_on_train=serialized_cols,
                )
                session.add(new_model)
                session.commit()
                return new_model.id

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
