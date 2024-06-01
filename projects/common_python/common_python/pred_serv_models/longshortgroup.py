from typing import Dict
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import LongShortGroup, Session


class LongShortGroupQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = LongShortGroup(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def get_strategies():
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(LongShortGroup)
                    .filter(LongShortGroup.is_disabled == False)
                    .all()
                )

    @staticmethod
    def get_all_strategies():
        with LogExceptionContext():
            with Session() as session:
                return session.query(LongShortGroup).all()

    @staticmethod
    def update(id, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                update_fields.pop("id", None)
                non_null_update_fields = {
                    k: v for k, v in update_fields.items() if v is not None
                }
                session.query(LongShortGroup).filter(LongShortGroup.id == id).update(
                    non_null_update_fields, synchronize_session=False
                )
                session.commit()

    @staticmethod
    def get_by_id(id: int):
        with LogExceptionContext():
            with Session() as session:
                entry = (
                    session.query(LongShortGroup)
                    .filter(LongShortGroup.id == id)
                    .first()
                )
                return entry

    @staticmethod
    def get_by_name(name: str):
        with LogExceptionContext():
            with Session() as session:
                entry = (
                    session.query(LongShortGroup)
                    .filter(LongShortGroup.name == name)
                    .first()
                )
                return entry
