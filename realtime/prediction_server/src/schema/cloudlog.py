from typing import Dict
from log import LogExceptionContext
from sqlalchemy import Column, DateTime, Integer, String, func
from orm import Base, Session

from datetime import datetime, timedelta


class LogLevels:
    EXCEPTION = "exception"
    INFO = "info"
    SYS = "system"
    DEBUG = "debug"


class CloudLog(Base):
    __tablename__ = "cloud_log"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    message = Column(String)
    level = Column(String)


class CloudLogQuery:
    @staticmethod
    def create_log_entry(fields: Dict):
        with LogExceptionContext():
            try:
                with Session() as session:
                    log_entry = CloudLog(**fields)
                    session.add(log_entry)
                    session.commit()
                    return log_entry.id
            except Exception:
                return "Error while creating log entry"

    @staticmethod
    def get_logs():
        with LogExceptionContext():
            with Session() as session:
                return session.query(CloudLog).all()

    @staticmethod
    def get_logs_by_level(level: str):
        with LogExceptionContext():
            with Session() as session:
                return session.query(CloudLog).filter(CloudLog.level == level).all()

    @staticmethod
    def delete_logs_older_than(date: DateTime):
        with LogExceptionContext():
            with Session() as session:
                session.query(CloudLog).filter(CloudLog.created_at < date).delete()
                session.commit()

    @staticmethod
    def clear_outdated_logs():
        ninety_days_ago = datetime.now() - timedelta(days=90)
        with LogExceptionContext():
            with Session() as session:
                session.query(CloudLog).filter(
                    CloudLog.created_at < ninety_days_ago
                ).delete()
                session.commit()

    @staticmethod
    def get_recent_logs(time_delta: timedelta):
        cutoff_date = datetime.now() - time_delta
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(CloudLog)
                    .filter(CloudLog.created_at >= cutoff_date)
                    .all()
                )
