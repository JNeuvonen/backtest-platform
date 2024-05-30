from typing import Dict
from datetime import timedelta

from sqlalchemy import Column, DateTime, Float, Integer, func
from common_python.pred_serv_orm import Base, Session
from common_python.log import LogExceptionContext


class BalanceSnapshot(Base):
    __tablename__ = "balance_snapshot"

    id = Column(Integer, primary_key=True)

    created_at = Column(DateTime, default=func.now())

    value = Column(Float)
    debt = Column(Float)
    long_assets_value = Column(Float)
    margin_level = Column(Float)
    btc_price = Column(Float)

    num_long_positions = Column(Integer)
    num_short_positions = Column(Integer)
    num_ls_positions = Column(Integer)


class BalanceSnapshotQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                account = BalanceSnapshot(**fields)
                session.add(account)
                session.commit()
                return account.id

    @staticmethod
    def update_entry(account_id: int, fields: Dict) -> bool:
        with LogExceptionContext():
            with Session() as session:
                account = session.query(BalanceSnapshot).filter_by(id=account_id).one()
                for key, value in fields.items():
                    setattr(account, key, value)
                session.commit()
                return True

    @staticmethod
    def delete_entry(account_id: int) -> bool:
        with LogExceptionContext():
            with Session() as session:
                account = session.query(BalanceSnapshot).filter_by(id=account_id).one()
                session.delete(account)
                session.commit()
                return True

    @staticmethod
    def read_all_entries():
        with LogExceptionContext():
            with Session() as session:
                accounts = (
                    session.query(BalanceSnapshot)
                    .order_by(BalanceSnapshot.id.asc())
                    .all()
                )
                return accounts

    @staticmethod
    def get_last_snapshot():
        with LogExceptionContext():
            with Session() as session:
                last_snapshot = (
                    session.query(BalanceSnapshot)
                    .order_by(BalanceSnapshot.created_at.desc())
                    .first()
                )
                return last_snapshot

    @staticmethod
    def get_snapshots_with_one_day_interval():
        with LogExceptionContext():
            with Session() as session:
                snapshots = (
                    session.query(BalanceSnapshot)
                    .order_by(BalanceSnapshot.created_at.asc())
                    .all()
                )

                result = []
                last_date = None

                for snapshot in snapshots:
                    if (
                        last_date is None
                        or snapshot.created_at >= last_date + timedelta(days=1)
                    ):
                        result.append(snapshot)
                        last_date = snapshot.created_at

                return result
