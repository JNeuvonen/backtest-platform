import threading
from typing import Dict
from sqlalchemy import Column, DateTime, Integer, String, func
from orm import Base, Session

from datetime import datetime, timedelta

from constants import LogLevel, LogSourceProgram, SlackWebhooks
from slack import post_slack_message
from schema.slack_bots import SlackWebhookQuery


class LogLevels:
    EXCEPTION = "exception"
    INFO = "info"
    SYS = "system"
    DEBUG = "debug"


class CloudLog(Base):
    __tablename__ = "cloud_log"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now())

    message = Column(String)
    level = Column(String)
    source_program = Column(Integer)


class CloudLogQuery:
    @staticmethod
    def create_log_entry(fields: Dict):
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
        with Session() as session:
            return session.query(CloudLog).all()

    @staticmethod
    def get_logs_by_level(level: str):
        with Session() as session:
            return session.query(CloudLog).filter(CloudLog.level == level).all()

    @staticmethod
    def delete_logs_older_than(date: DateTime):
        with Session() as session:
            session.query(CloudLog).filter(CloudLog.created_at < date).delete()
            session.commit()

    @staticmethod
    def clear_outdated_logs():
        ninety_days_ago = datetime.now() - timedelta(days=3)
        with Session() as session:
            session.query(CloudLog).filter(
                CloudLog.created_at < ninety_days_ago
            ).delete()
            session.commit()

    @staticmethod
    def get_recent_logs(time_delta: timedelta):
        cutoff_date = datetime.now() - time_delta
        with Session() as session:
            return (
                session.query(CloudLog).filter(CloudLog.created_at >= cutoff_date).all()
            )


def slack_log(msg: str, source_program: int, level: str):
    def func_helper():
        if LogSourceProgram.PRED_SERVER == source_program:
            all_channel_hook = SlackWebhookQuery.get_webhook_by_name(
                SlackWebhooks.PRED_SERV_ALL
            )

            if LogLevel.EXCEPTION == level:
                exception_channel_hook = SlackWebhookQuery.get_webhook_by_name(
                    SlackWebhooks.PRED_SERV_EXCEPTIONS
                )

                if exception_channel_hook is not None:
                    post_slack_message(exception_channel_hook.webhook_uri, msg)

            if all_channel_hook is not None:
                post_slack_message(all_channel_hook.webhook_uri, msg)

        if LogSourceProgram.TRADING_CLIENT == source_program:
            all_channel_hook = SlackWebhookQuery.get_webhook_by_name(
                SlackWebhooks.TRADE_CLIENT_ALL
            )

            if LogLevel.EXCEPTION == level:
                exception_channel_hook = SlackWebhookQuery.get_webhook_by_name(
                    SlackWebhooks.TRADE_CLIENT_EXCEPTIONS
                )

                if exception_channel_hook is not None:
                    post_slack_message(exception_channel_hook.webhook_uri, msg)

            if all_channel_hook is not None:
                post_slack_message(all_channel_hook.webhook_uri, msg)

    thread = threading.Thread(target=func_helper)
    thread.start()


def create_log(msg: str, level: str, slack_bot=None):
    if slack_bot is not None:
        slack_webhook = SlackWebhookQuery.get_webhook_by_name(slack_bot)

        if slack_webhook is not None:
            slack_log(msg, LogSourceProgram.PRED_SERVER, level)

    CloudLogQuery.create_log_entry(
        {
            "message": msg,
            "level": level,
            "source_program": LogSourceProgram.PRED_SERVER,
        }
    )
