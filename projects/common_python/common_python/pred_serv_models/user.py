from typing import Dict
from sqlalchemy import Column, DateTime, Integer, String, func
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Base, Session


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    first_name = Column(String)
    last_name = Column(String)
    email = Column(String)

    access_level = Column(Integer)


class UserQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = User(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def get_users():
        with LogExceptionContext():
            with Session() as session:
                return session.query(User).all()

    @staticmethod
    def update_user(user_id: int, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                update_fields.pop("id", None)
                non_null_update_fields = {
                    k: v for k, v in update_fields.items() if v is not None
                }
                session.query(User).filter(User.id == user_id).update(
                    non_null_update_fields, synchronize_session="fetch"
                )
                session.commit()

    @staticmethod
    def get_user_by_id(user_id: int):
        with LogExceptionContext():
            with Session() as session:
                return session.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_user_by_email(email: str):
        with LogExceptionContext():
            with Session() as session:
                return session.query(User).filter(User.email == email).first()
