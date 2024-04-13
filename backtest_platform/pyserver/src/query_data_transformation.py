from typing import Dict
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from log import LogExceptionContext
from orm import Base, Session


class DataTransformation(Base):
    __tablename__ = "data_transformation"
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("dataset.id"), nullable=False)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    transformation_code = Column(String, nullable=False)


class DataTransformationQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            try:
                with Session() as session:
                    transformation = DataTransformation(**fields)
                    session.add(transformation)
                    session.commit()
                    return transformation.id
            except Exception as e:
                return str(e)

    @staticmethod
    def get_transformations():
        with LogExceptionContext():
            with Session() as session:
                return session.query(DataTransformation).all()

    @staticmethod
    def remove_transformation_by_id(transformation_id: int):
        with LogExceptionContext():
            with Session() as session:
                session.query(DataTransformation).filter(
                    DataTransformation.id == transformation_id
                ).delete()
                session.commit()

    @staticmethod
    def get_transformations_by_dataset(dataset_id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(DataTransformation)
                    .filter(DataTransformation.dataset_id == dataset_id)
                    .order_by(DataTransformation.id)
                    .all()
                )
