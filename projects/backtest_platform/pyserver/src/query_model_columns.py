import json
from typing import List

from sqlalchemy import Column, ForeignKey, Integer, String
from log import LogExceptionContext
from orm import Base, Session


class ModelColumns(Base):
    __tablename__ = "mode_columns"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("model.id"))
    columns = Column(String)


class ModelColumnsQuery:
    @staticmethod
    def create_columns_entry(model_id: int, columns: List[str]):
        with LogExceptionContext():
            with Session() as session:
                columns_serialized = json.dumps(columns)
                model_columns = ModelColumns(
                    model_id=model_id, columns=columns_serialized
                )
                session.add(model_columns)
                session.commit()
                return model_columns.id

    @staticmethod
    def get_columns_by_model_id(model_id: int):
        with LogExceptionContext():
            with Session() as session:
                model_columns = (
                    session.query(ModelColumns)
                    .filter(ModelColumns.model_id == model_id)
                    .one_or_none()
                )
                if model_columns:
                    return json.loads(model_columns.columns)
                return None

    @staticmethod
    def update_columns_by_model_id(model_id: int, new_columns: List[str]):
        with LogExceptionContext():
            with Session() as session:
                columns_serialized = json.dumps(new_columns)
                session.query(ModelColumns).filter(
                    ModelColumns.model_id == model_id
                ).update({"columns": columns_serialized})
                session.commit()

    @staticmethod
    def delete_columns_by_model_id(model_id: int):
        with LogExceptionContext():
            with Session() as session:
                session.query(ModelColumns).filter(
                    ModelColumns.model_id == model_id
                ).delete()
                session.commit()
