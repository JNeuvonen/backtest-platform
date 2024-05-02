from typing import Dict, List
from sqlalchemy import Column, Integer, String, UniqueConstraint
from log import LogExceptionContext
from orm import Base, Session


class CodePreset(Base):
    __tablename__ = "code_preset"
    id = Column(Integer, primary_key=True)
    code = Column(String)
    category = Column(String)
    name = Column(String)
    description = Column(String)
    label = Column(String)
    __table_args__ = (UniqueConstraint("category", "name", name="_category_name_uc"),)


class CodePresetQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = CodePreset(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def retrieve_all(preset_id: str):
        with LogExceptionContext():
            with Session() as session:
                code_presets = (
                    session.query(CodePreset)
                    .filter(CodePreset.category == preset_id)
                    .all()
                )
                return code_presets

    @staticmethod
    def fetch_one_by_id(id: int) -> CodePreset:
        with LogExceptionContext():
            with Session() as session:
                code_preset = (
                    session.query(CodePreset).filter(CodePreset.id == id).one_or_none()
                )
                return code_preset

    @staticmethod
    def fetch_all():
        with LogExceptionContext():
            with Session() as session:
                return session.query(CodePreset).all()

    @staticmethod
    def create_many(entries: List[Dict]):
        successful_ids = []
        with LogExceptionContext():
            with Session() as session:
                for fields in entries:
                    try:
                        entry = CodePreset(**fields)
                        session.add(entry)
                        session.commit()
                        successful_ids.append(entry.id)
                    except Exception:
                        session.rollback()
                return successful_ids

    @staticmethod
    def update_entry(preset_id: int, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = (
                    session.query(CodePreset).filter(CodePreset.id == preset_id).one()
                )
                for key, value in update_fields.items():
                    setattr(entry, key, value)
                session.commit()

    @staticmethod
    def delete_by_id(preset_id: int):
        with LogExceptionContext():
            with Session() as session:
                entry = (
                    session.query(CodePreset).filter(CodePreset.id == preset_id).one()
                )
                session.delete(entry)
                session.commit()
