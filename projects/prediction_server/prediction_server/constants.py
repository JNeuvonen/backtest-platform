class LogLevel:
    EXCEPTION = "exception"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


class LogSourceProgram:
    TRADING_CLIENT = 1
    PRED_SERVER = 2


class SlackWebhooks:
    ALL_LOG_BOT = "log_bot_all"

    PRED_SERV_EXCEPTIONS = "pred_serv_exceptions"
    PRED_SERV_ALL = "pred_serv_all"

    TRADE_CLIENT_EXCEPTIONS = "trade_client_exceptions"
    TRADE_CLIENT_ALL = "trade_client_all"

    TRADE_NOTIFS = "trade_notifications"


def strategy_name_to_db_file(strat_name: str):
    return strat_name + ".db"


KLINES_MAX_TIME_RANGE = "1 Jan, 2017"
SOFTWARE_VERSION = "1.0.1"


class TradeDirection:
    LONG = "LONG"
    SHORT = "SHORT"
