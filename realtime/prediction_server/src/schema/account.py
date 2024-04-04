from typing import Dict
from orm import Base, Session
from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy import DateTime
from sqlalchemy.sql import func

from log import LogExceptionContext


class Account(Base):
    __tablename__ = "account"

    id = Column(Integer, primary_key=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    name = Column(String, unique=True)
    max_debt_ratio = Column(Float, default=0.0)


class AccountQuery:
    @staticmethod
    def create_account(fields: Dict):
        with LogExceptionContext(re_raise=False):
            with Session() as session:
                account = Account(**fields)
                session.add(account)
                session.commit()
                return account.id

    @staticmethod
    def get_accounts():
        with LogExceptionContext():
            with Session() as session:
                return session.query(Account).all()

    @staticmethod
    def update_account(id: int, fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                session.query(Account).filter(Account.id == id).update(fields)
                session.commit()

    @staticmethod
    def delete_account(id: int):
        with LogExceptionContext():
            with Session() as session:
                session.query(Account).filter(Account.id == id).delete()
                session.commit()

    @staticmethod
    def get_account_by_id(id: int):
        with LogExceptionContext():
            with Session() as session:
                return session.query(Account).filter(Account.id == id).first()

    @staticmethod
    def get_account_by_name(name: str):
        with LogExceptionContext():
            with Session() as session:
                return session.query(Account).filter(Account.name == name).first()
