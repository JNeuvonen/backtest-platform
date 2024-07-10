from typing import Dict, List
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from log import LogExceptionContext
from orm import Base, Session


class DatasetCombine(Base):
    __tablename__ = "dataset_combine"
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    source_dataset_id = Column(Integer, ForeignKey("dataset.id"))
    source_dataset_table_name = Column(String)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    added_columns = Column(String)


class DatasetCombineQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            try:
                with Session() as session:
                    transformation = DatasetCombine(**fields)
                    session.add(transformation)
                    session.commit()
                    return transformation.id
            except Exception as e:
                return str(e)

    @staticmethod
    def fetch_by_dataset_id(dataset_id: int):
        with LogExceptionContext():
            try:
                with Session() as session:
                    results = (
                        session.query(DatasetCombine)
                        .filter_by(dataset_id=dataset_id)
                        .all()
                    )
                    return results
            except Exception as e:
                return str(e)
