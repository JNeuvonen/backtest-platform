from typing import Dict
from sqlalchemy import Column, Integer, String
from log import LogExceptionContext
from orm import Base, Session


class CodePreset(Base):
    __tablename__ = "code_preset"
    id = Column(Integer, primary_key=True)
    code = Column(String)
    category = Column(String)
    name = Column(String)


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
