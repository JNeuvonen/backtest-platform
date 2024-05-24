from typing import Dict
from sqlalchemy import Column, DateTime, Integer, String, func
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Base, Session


class WhiteListedIP(Base):
    __tablename__ = "whitelisted_ip"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    ip = Column(String, unique=True)


class WhiteListedIPQuery:
    @staticmethod
    def create_entry(fields: Dict):
        try:
            with Session() as session:
                entry = WhiteListedIP(**fields)
                session.add(entry)
                session.commit()
                return entry.id
        except Exception:
            return "Unique constraint failed"

    @staticmethod
    def get_whitelisted_ips():
        with LogExceptionContext():
            with Session() as session:
                return session.query(WhiteListedIP).all()

    @staticmethod
    def remove_whitelisted_ip_by_ip(ip: str):
        with LogExceptionContext():
            with Session() as session:
                session.query(WhiteListedIP).filter(WhiteListedIP.ip == ip).delete()
                session.commit()

    @staticmethod
    def is_allowed_ip(ip: str) -> bool:
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(WhiteListedIP).filter(WhiteListedIP.ip == ip).first()
                    is not None
                )
