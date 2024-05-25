import threading
from typing import Dict
from sqlalchemy import Column, DateTime, Integer, String, func

from common_python.pred_serv_orm import Base, Session
from datetime import datetime, timedelta

from common_python.constants import (
    LogLevel,
    LogSourceProgram,
    SlackWebhooks,
    TradeDirection,
)
from common_python.pred_serv_models.slack_bots import SlackWebhookQuery
from common_python.slack import post_slack_message


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


def create_trade_enter_notif_msg(body) -> str:
    direction_emoji = "📈" if body.direction == TradeDirection.LONG else "📉"
    return (
        f"```Trade Alert - {body.symbol} {direction_emoji}```\n"
        f"------------\n"
        f"- Quantity: `{body.quantity}` at `{body.open_price}`\n"
        f"- Opened at: `{datetime.fromtimestamp(body.open_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')}`"
    )


def create_ls_trade_close_notif_msg(info_dict: dict) -> str:
    close_time = datetime.fromtimestamp(info_dict["close_time_ms"] / 1000).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    open_time = datetime.fromtimestamp(info_dict["open_time_ms"] / 1000).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    hold_time_seconds = (info_dict["close_time_ms"] - info_dict["open_time_ms"]) / 1000
    if hold_time_seconds < 86400:
        hold_time = f"{int(hold_time_seconds / 60)} minutes"
    else:
        hold_time = f"{int(hold_time_seconds / 86400)} days"

    return (
        f"```Long/Short Trade Close Alert```"
        f"\n------------"
        f"\n- Long Side Symbol: `{info_dict['long_side_symbol']}`"
        f"\n- Short Side Symbol: `{info_dict['short_side_symbol']}`"
        f"\n- Hold Time: `{hold_time}`"
        f"\n- Opened at: `{open_time}`"
        f"\n- Closed at: `{close_time}`"
        f"\n- Net Result: `{info_dict['net_result']}`"
        f"\n- Percent Result: `{info_dict['combined_perc_result']}%`"
        f"\n- Long Side Net Result: `{info_dict['long_side_net_result']}`"
        f"\n- Short Side Net Result: `{info_dict['short_side_net_result']}`"
        f"\n- Long Side Percent Result: `{info_dict['long_side_perc_result']}%`"
        f"\n- Short Side Percent Result: `{info_dict['short_side_perc_result']}%`"
    )


def create_trade_close_notif_msg(strat, trade_update_dict: dict) -> str:
    direction_emoji = "📉" if strat.is_short_selling_strategy else "📈"
    close_time = datetime.fromtimestamp(
        trade_update_dict["close_time_ms"] / 1000
    ).strftime("%Y-%m-%d %H:%M:%S")
    open_time = datetime.fromtimestamp(strat.time_on_trade_open_ms / 1000).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    hold_time_seconds = (
        trade_update_dict["close_time_ms"] - strat.time_on_trade_open_ms
    ) / 1000
    if hold_time_seconds < 86400:
        hold_time = f"{int(hold_time_seconds / 60)} minutes"
    else:
        hold_time = f"{int(hold_time_seconds / 86400)} days"
    return (
        f"```Trade Close Alert - {strat.symbol} {direction_emoji}```\n"
        f"------------\n"
        f"- Quantity: `{strat.quantity_on_trade_open}` at `{trade_update_dict['close_price']}`\n"
        f"- Hold Time: `{hold_time}`\n"
        f"- Opened at: `{open_time}`\n"
        f"- Closed at: `{close_time}`\n"
        f"- Net Result: `{trade_update_dict.get('net_result', 'N/A')}`\n"
        f"- Percent Result: `{trade_update_dict.get('percent_result', 'N/A')}%`"
    )


def slack_log_close_trade_notif(strat, trade_update_dict):
    def func_helper():
        hook = SlackWebhookQuery.get_webhook_by_name(SlackWebhooks.TRADE_NOTIFS)

        if hook is None:
            create_log(
                "Hook for TRADE_NOTIFS is not correctly set", level=LogLevel.EXCEPTION
            )
            return
        post_slack_message(
            hook.webhook_uri, create_trade_close_notif_msg(strat, trade_update_dict)
        )

    thread = threading.Thread(target=func_helper)
    thread.start()


def slack_log_close_ls_trade_notif(info_dict):
    def func_helper():
        hook = SlackWebhookQuery.get_webhook_by_name(SlackWebhooks.TRADE_NOTIFS)

        if hook is None:
            create_log(
                "Hook for TRADE_NOTIFS is not correctly set", level=LogLevel.EXCEPTION
            )
            return

        post_slack_message(hook.webhook_uri, create_ls_trade_close_notif_msg(info_dict))

    thread = threading.Thread(target=func_helper)
    thread.start()


def slack_log_enter_trade_notif(body):
    def func_helper():
        hook = SlackWebhookQuery.get_webhook_by_name(SlackWebhooks.TRADE_NOTIFS)

        if hook is None:
            create_log(
                "Hook for TRADE_NOTIFS is not correctly set", level=LogLevel.EXCEPTION
            )
            return

        post_slack_message(hook.webhook_uri, create_trade_enter_notif_msg(body))

    thread = threading.Thread(target=func_helper)
    thread.start()


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

        if LogSourceProgram.ANALYTICS_SERVER == source_program:
            all_channel_hook = SlackWebhookQuery.get_webhook_by_name(
                SlackWebhooks.ANALYTICS_ALL
            )

            if LogLevel.EXCEPTION == level:
                exception_channel_hook = SlackWebhookQuery.get_webhook_by_name(
                    SlackWebhooks.ANALYTICS_EXCEPTIONS
                )
                if exception_channel_hook is not None:
                    post_slack_message(exception_channel_hook.webhook_uri, msg)

            if all_channel_hook is not None:
                post_slack_message(all_channel_hook.webhook_uri, msg)

    thread = threading.Thread(target=func_helper)
    thread.start()


def create_log(msg: str, level: str, source_program=LogSourceProgram.PRED_SERVER):
    slack_log(msg, source_program, level)

    def func_helper():
        CloudLogQuery.create_log_entry(
            {
                "message": msg,
                "level": level,
                "source_program": source_program,
            }
        )

    thread = threading.Thread(target=func_helper)
    thread.start()
