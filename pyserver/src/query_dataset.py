from sqlalchemy import Column, Integer, String, update
from log import LogExceptionContext
from orm import Base, Session


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True)
    dataset_name = Column(String)
    timeseries_column = Column(String)
    target_column = Column(String)


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

    @staticmethod
    def get_target_col(dataset_name: str):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(Dataset.target_column)
                    .filter(Dataset.dataset_name == dataset_name)
                    .scalar()
                )