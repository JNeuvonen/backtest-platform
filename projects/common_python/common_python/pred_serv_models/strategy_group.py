import json
from typing import Dict

from sqlalchemy import func
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Session, StrategyGroup


class StrategyGroupQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                # Serialize the transformation_ids list to a JSON string
                if "transformation_ids" in fields and isinstance(
                    fields["transformation_ids"], list
                ):
                    fields["transformation_ids"] = json.dumps(
                        fields["transformation_ids"]
                    )
                entry = StrategyGroup(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def get_all():
        with LogExceptionContext():
            with Session() as session:
                results = session.query(StrategyGroup).all()
                for result in results:
                    if result.transformation_ids:
                        result.transformation_ids = json.loads(
                            result.transformation_ids
                        )
                return results

    @staticmethod
    def get_all_active():
        with LogExceptionContext():
            with Session() as session:
                results = (
                    session.query(StrategyGroup)
                    .filter(StrategyGroup.is_disabled == False)
                    .all()
                )
                for result in results:
                    if result.transformation_ids:
                        result.transformation_ids = json.loads(
                            result.transformation_ids
                        )
                return results

    @staticmethod
    def update(strategy_group_id: int, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                update_fields.pop("id", None)
                # Serialize the transformation_ids list to a JSON string if it's in update_fields
                if "transformation_ids" in update_fields and isinstance(
                    update_fields["transformation_ids"], list
                ):
                    update_fields["transformation_ids"] = json.dumps(
                        update_fields["transformation_ids"]
                    )
                non_null_update_fields = {
                    k: v for k, v in update_fields.items() if v is not None
                }
                session.query(StrategyGroup).filter(
                    StrategyGroup.id == strategy_group_id
                ).update(non_null_update_fields, synchronize_session=False)
                session.commit()

    @staticmethod
    def get_by_name(name: str):
        with LogExceptionContext():
            with Session() as session:
                result = (
                    session.query(StrategyGroup)
                    .filter(StrategyGroup.name == name)
                    .first()
                )
                if result and result.transformation_ids:
                    result.transformation_ids = json.loads(result.transformation_ids)
                return result

    @staticmethod
    def get_by_id(strategy_group_id: int):
        with LogExceptionContext():
            with Session() as session:
                result = (
                    session.query(StrategyGroup)
                    .filter(StrategyGroup.id == strategy_group_id)
                    .first()
                )
                if result and result.transformation_ids:
                    result.transformation_ids = json.loads(result.transformation_ids)
                return result

    @staticmethod
    def update_last_adaptive_group_recalc(strategy_group_id: int):
        with LogExceptionContext():
            with Session() as session:
                session.query(StrategyGroup).filter(
                    StrategyGroup.id == strategy_group_id
                ).update(
                    {"last_adaptive_group_recalc": func.now()},
                    synchronize_session=False,
                )
                session.commit()
