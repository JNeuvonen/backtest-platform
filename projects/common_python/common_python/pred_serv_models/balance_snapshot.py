from typing import Dict

from sqlalchemy import Column, DateTime, Float, Integer, func
from common_python.pred_serv_orm import Base, Session


class BalanceSnapshot(Base):
    __tablename__ = "balance_snapshot"

    id = Column(Integer, primary_key=True)

    created_at = Column(DateTime, default=func.now())

    value = Column(Float)
    debt = Column(Float)
    long_assets_value = Column(Float)
    margin_level = Column(Float)

    num_directional_positions = Column(Integer)
    num_ls_positions = Column(Integer)


class BalanceSnapshotQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with Session() as session:
            account = BalanceSnapshot(**fields)
            session.add(account)
            session.commit()
            return account.id

    @staticmethod
    def update_entry(account_id: int, fields: Dict) -> bool:
        with Session() as session:
            account = session.query(BalanceSnapshot).filter_by(id=account_id).one()
            for key, value in fields.items():
                setattr(account, key, value)
            session.commit()
            return True

    @staticmethod
    def delete_entry(account_id: int) -> bool:
        with Session() as session:
            account = session.query(BalanceSnapshot).filter_by(id=account_id).one()
            session.delete(account)
            session.commit()
            return True

    @staticmethod
    def read_all_entries():
        with Session() as session:
            accounts = session.query(BalanceSnapshot).all()
            return accounts
