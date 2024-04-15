from typing import Dict
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from orm import Base, Session
from log import LogExceptionContext


class DataTransformation(Base):
    __tablename__ = "data_transformation"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategy.id"), nullable=False)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    transformation_code = Column(String, nullable=False)


class DataTransformationQuery:
    @staticmethod
    def create_transformation(fields: Dict):
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
    def get_transformations_by_strategy(strategy_id):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(DataTransformation)
                    .filter(DataTransformation.strategy_id == strategy_id)
                    .order_by(DataTransformation.id)
                    .all()
                )
