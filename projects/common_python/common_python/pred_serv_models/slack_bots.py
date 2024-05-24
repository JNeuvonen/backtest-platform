from typing import Dict
from sqlalchemy import Column, DateTime, Integer, String, func
from common_python.log import LogExceptionContext
from common_python.pred_serv_orm import Base, Session


class SlackWebhook(Base):
    __tablename__ = "slackbot"

    id = Column(Integer, primary_key=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    name = Column(String, unique=True)
    webhook_uri = Column(String, nullable=False)


class SlackWebhookQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = SlackWebhook(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def get_webhooks():
        with LogExceptionContext():
            with Session() as session:
                return session.query(SlackWebhook).all()

    @staticmethod
    def update_webhook(webhook_id, update_fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                update_fields.pop("id", None)
                non_null_update_fields = {
                    k: v for k, v in update_fields.items() if v is not None
                }
                session.query(SlackWebhook).filter(
                    SlackWebhook.id == webhook_id
                ).update(non_null_update_fields, synchronize_session="fetch")
                session.commit()

    @staticmethod
    def get_webhook_by_id(webhook_id: int):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(SlackWebhook)
                    .filter(SlackWebhook.id == webhook_id)
                    .first()
                )

    @staticmethod
    def get_webhook_by_name(webhook_name: str):
        with LogExceptionContext():
            with Session() as session:
                return (
                    session.query(SlackWebhook)
                    .filter(SlackWebhook.name == webhook_name)
                    .first()
                )
