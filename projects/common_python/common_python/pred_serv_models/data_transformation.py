from typing import Dict
from common_python.pred_serv_orm import Session
from common_python.pred_serv_models.datatransformation import DataTransformation


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
    def get_transformations():
        with Session() as session:
            return session.query(DataTransformation).all()

    @staticmethod
    def remove_transformation_by_id(transformation_id: int):
        with Session() as session:
            session.query(DataTransformation).filter(
                DataTransformation.id == transformation_id
            ).delete()
            session.commit()

    @staticmethod
    def get_transformations_by_strategy(strategy_id):
        with Session() as session:
            return (
                session.query(DataTransformation)
                .filter(DataTransformation.strategy_id == strategy_id)
                .order_by(DataTransformation.id)
                .all()
            )

    @staticmethod
    def get_transformations_by_longshort_group(longshort_group_id):
        with Session() as session:
            return (
                session.query(DataTransformation)
                .filter(DataTransformation.long_short_group_id == longshort_group_id)
                .order_by(DataTransformation.id)
                .all()
            )
