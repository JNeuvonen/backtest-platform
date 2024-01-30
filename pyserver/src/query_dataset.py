from typing import List
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, update
from log import LogExceptionContext
from orm import Base, Session


class DatasetBody(BaseModel):
    id: int
    dataset_name: str
    timeseries_column: str
    price_column: str
    target_column: str


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True)
    dataset_name = Column(String, unique=True)
    timeseries_column = Column(String)
    price_column = Column(String)
    target_column = Column(String)

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_name": self.dataset_name,
            "timeseries_column": self.timeseries_column,
            "price_column": self.price_column,
            "target_column": self.target_column,
        }


class DatasetQuery:
    @staticmethod
    def update_dataset(dataset_name: str, update_fields: DatasetBody):
        with LogExceptionContext():
            with Session() as session:
                session.execute(
                    update(Dataset)
                    .where(Dataset.dataset_name == dataset_name)
                    .values(**update_fields.model_dump())
                )
                session.commit()

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
    def delete_entry_by_dataset_name(dataset_name: str):
        with LogExceptionContext():
            with Session() as session:
                session.query(Dataset).filter(
                    Dataset.dataset_name == dataset_name
                ).delete()
                session.commit()

    @staticmethod
    def update_target_column(dataset_name: str, target_column: str):
        with LogExceptionContext():
            with Session() as session:
                session.execute(
                    update(Dataset)
                    .where(Dataset.dataset_name == dataset_name)
                    .values(target_column=target_column)
                )
                session.commit()

    @staticmethod
    def create_dataset_entry(
        dataset_name: str,
        timeseries_column: str,
        target_column: str | None = None,
        price_column: str | None = None,
    ):
        with LogExceptionContext():
            with Session() as session:
                new_dataset = Dataset(
                    dataset_name=dataset_name,
                    timeseries_column=timeseries_column,
                    target_column=target_column,
                    price_column=price_column,
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

    @staticmethod
    def get_target_col(dataset_name: str):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(Dataset.target_column)
                    .filter(Dataset.dataset_name == dataset_name)
                    .scalar()
                )

    @staticmethod
    def get_price_col(dataset_name: str):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(Dataset.price_column)
                    .filter(Dataset.dataset_name == dataset_name)
                    .scalar()
                )

    @staticmethod
    def fetch_dataset_by_name(dataset_name: str):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(Dataset)
                    .filter(Dataset.dataset_name == dataset_name)
                    .first()
                )

    @staticmethod
    def update_price_column(dataset_name: str, new_price_column: str):
        with LogExceptionContext():
            with Session() as session:
                session.execute(
                    update(Dataset)
                    .where(Dataset.dataset_name == dataset_name)
                    .values(price_column=new_price_column)
                )
                session.commit()
