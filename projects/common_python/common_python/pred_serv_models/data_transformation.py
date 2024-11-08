from typing import Dict

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, and_, func
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Base, Session


class DataTransformation(Base):
    __tablename__ = "data_transformation"
    id = Column(Integer, primary_key=True)
    long_short_group_id = Column(Integer, ForeignKey("long_short_group.id"))
    strategy_id = Column(Integer, ForeignKey("strategy.id"))
    strategy_group_id = Column(Integer, ForeignKey("strategy_group.id"))

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    transformation_code = Column(String, nullable=False)


class DataTransformationQuery:
    @staticmethod
    def create_transformation(fields: Dict):
        try:
            with Session() as session:
                transformation = DataTransformation(**fields)
                session.add(transformation)
                session.commit()
                return transformation.id
        except Exception as e:
            return str(e)

    @staticmethod
    def update(strategy_id, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                update_fields.pop("id", None)
                non_null_update_fields = {
                    k: v for k, v in update_fields.items() if v is not None
                }
                session.query(DataTransformation).filter(
                    DataTransformation.id == strategy_id
                ).update(non_null_update_fields, synchronize_session=False)
                session.commit()

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

    @staticmethod
    def get_transformations_by_longshort_group(longshort_group_id):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(DataTransformation)
                    .filter(
                        DataTransformation.long_short_group_id == longshort_group_id
                    )
                    .order_by(DataTransformation.id)
                    .all()
                )

    @staticmethod
    def get_all_strategy_group_transformations():
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(DataTransformation)
                    .filter(DataTransformation.strategy_group_id != None)
                    .order_by(DataTransformation.id)
                    .all()
                )

    @staticmethod
    def delete_transformations_not_in_list(ids: list):
        try:
            with Session() as session:
                session.query(DataTransformation).filter(
                    and_(
                        DataTransformation.id.notin_(ids),
                        DataTransformation.long_short_group_id == None,
                    )
                ).delete(synchronize_session=False)
                session.commit()
        except Exception as e:
            return str(e)
