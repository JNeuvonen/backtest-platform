import secrets
from typing import Dict
from common_python.log import LogExceptionContext
from sqlalchemy import Column, DateTime, Integer, String, func
from common_python.pred_serv_orm import Base, Session


class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    key = Column(String, unique=True)


def generate_api_key():
    return secrets.token_urlsafe(32)


class APIKeyQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            try:
                with Session() as session:
                    entry = APIKey(**fields)
                    session.add(entry)
                    session.commit()
                    return entry.id
            except Exception:
                return "Unique constraint failed"

    @staticmethod
    def get_api_keys():
        with LogExceptionContext():
            with Session() as session:
                return session.query(APIKey).all()

    @staticmethod
    def remove_api_key_by_key(key: str):
        with LogExceptionContext():
            with Session() as session:
                session.query(APIKey).filter(APIKey.key == key).delete()
                session.commit()

    @staticmethod
    def gen_api_key():
        with LogExceptionContext():
            api_key = generate_api_key()
            entry = {"key": api_key}
            APIKeyQuery.create_entry(entry)
            return entry

    @staticmethod
    def is_valid_key(key: str) -> bool:
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(APIKey).filter(APIKey.key == key).first() is not None
                )
